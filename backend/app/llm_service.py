import os
import time
import random
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from fastapi import HTTPException

# Force LiteLLM to use local offline model cost map
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm.caching.caching import Cache
from litellm.exceptions import (
    RateLimitError,
    APIError,
    APIConnectionError,
    ServiceUnavailableError,
    InternalServerError,
    BadGatewayError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
)

from app.config import settings

logger = logging.getLogger("backend.llm")
litellm.telemetry = False
litellm.suppress_debug_info = True

# Exponential backoff schedule for temporary real-provider errors (per key)
# attempt 1 -> wait 1.0s + jitter, attempt 2 -> wait 2.0s + jitter, attempt 3 -> fail over to next key
BACKOFF_DELAYS: List[float] = [1.0, 2.0]
MAX_ATTEMPTS_PER_KEY: int = 3

# Initialize LiteLLM Cache
cache_type = settings.CACHE_TYPE if settings.CACHE_TYPE in ("local", "disk") else "local"
litellm.cache = Cache(type=cache_type, mode="default_on")
litellm.enable_cache(type=cache_type)


def is_cache_configured() -> bool:
    """Returns True if LiteLLM cache is configured."""
    return litellm.cache is not None


def is_permanent_error(exc: Exception) -> bool:
    """Identifies authentication, permission, or client errors that must not be retried."""
    if isinstance(exc, (AuthenticationError, PermissionDeniedError, NotFoundError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in (400, 401, 403, 404):
        return True
    resp = getattr(exc, "response", None)
    if resp and getattr(resp, "status_code", None) in (400, 401, 403, 404):
        return True
    return False


def is_gemini_quota_exhausted(exc: Exception) -> bool:
    """
    Identifies whether an exception represents a daily/project quota exhaustion
    (e.g., FreeTier limit, RESOURCE_EXHAUSTED, GenerateRequestsPerDayPerProjectPerModel-FreeTier).
    Such errors are NOT transient network spikes and must NOT be retried on the same key.
    """
    err_str = str(exc).lower()
    quota_signatures = [
        "resource_exhausted",
        "generaterequestsperday",
        "freetier",
        "free_tier",
        "quota exceeded",
        "quota_exceeded",
        "exceeded your current quota",
        "daily quota",
        "per day",
    ]
    return any(sig in err_str for sig in quota_signatures)


def is_retryable_error(exc: Exception) -> bool:
    """
    Identifies whether an exception represents a temporary, retryable error:
    - HTTP 503 Service Unavailable / High demand
    - Transient HTTP 429 RateLimit (NOT daily/project quota exhaustion)
    - Timeout / Read timed out
    - HTTP 502 Bad Gateway / HTTP 504 Gateway Timeout
    - APIConnectionError
    """
    if is_permanent_error(exc):
        return False

    # Never retry daily/project quota exhaustion repeatedly
    if is_gemini_quota_exhausted(exc):
        return False

    if isinstance(exc, (ServiceUnavailableError, litellm.exceptions.Timeout, APIConnectionError, BadGatewayError)):
        return True

    if isinstance(exc, RateLimitError):
        return not is_gemini_quota_exhausted(exc)

    status_code = getattr(exc, "status_code", None)
    if status_code in (502, 503, 504):
        return True
    if status_code == 429:
        return not is_gemini_quota_exhausted(exc)

    resp = getattr(exc, "response", None)
    if resp and getattr(resp, "status_code", None) in (502, 503, 504):
        return True
    if resp and getattr(resp, "status_code", None) == 429:
        return not is_gemini_quota_exhausted(exc)

    err_str = str(exc).lower()
    retryable_keywords = [
        "503", "timeout", "timed out", "unavailable",
        "high demand", "rate limit",
        "temporary", "connection reset", "connection refused", "bad gateway"
    ]
    if any(kw in err_str for kw in retryable_keywords):
        return not is_gemini_quota_exhausted(exc)

    return False


def is_http_503_error(exc: Exception) -> bool:
    """Backwards compatibility helper for existing 503 checks."""
    if isinstance(exc, ServiceUnavailableError):
        return True
    if getattr(exc, "status_code", None) == 503:
        return True
    resp = getattr(exc, "response", None)
    if resp and getattr(resp, "status_code", None) == 503:
        return True
    err_str = str(exc)
    if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
        return True
    return False


def parse_simulate_failure_header(header_val: Optional[str]) -> Dict[str, int]:
    """
    Parses X-Simulate-Failure header.
    Format: comma-separated <key_id>=<status> pairs; * means every key.
    Supported statuses: 429, 500, 502, 503.
    """
    if not header_val:
        return {}
    sim = {}
    parts = [p.strip() for p in header_val.split(",") if p.strip()]
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                code = int(v)
                if code in (429, 500, 502, 503):
                    sim[k] = code
            except ValueError:
                pass
    return sim


def calculate_tokens(model: str, messages: List[Dict[str, str]], reply_text: str) -> Tuple[int, int]:
    """Calculates input and output tokens accurately using LiteLLM."""
    try:
        input_tokens = litellm.token_counter(model=model, messages=messages)
    except Exception:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        input_tokens = max(1, total_chars // 4)

    try:
        output_tokens = litellm.token_counter(model=model, text=reply_text)
    except Exception:
        output_tokens = max(1, len(reply_text) // 4)

    return int(input_tokens), int(output_tokens)


class LLMService:
    def __init__(self):
        self.model = settings.normalized_model
        self.timeout = settings.LLM_REQUEST_TIMEOUT

    def get_configured_keys_count(self) -> int:
        return len(settings.key_pool)

    async def execute_chat(
        self,
        messages: List[Dict[str, str]],
        simulate_failure_header: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes chat completion with:
        1. Cache check (LiteLLM Cache)
        2. Automatic key pool failover (at least 2 keys)
        3. Real temporary errors (503, 429, timeout) exponential backoff retry with jitter per key
        4. Immediate failover for non-retryable, permanent, or simulated errors (NO retry on X-Simulate-Failure)
        5. Token tracking (input_tokens for this request, output_tokens for reply)
        6. Accurate latency measurement
        """
        start_time = time.perf_counter()
        sim_map = {}
        if settings.ENABLE_FAILURE_SIMULATION and simulate_failure_header:
            sim_map = parse_simulate_failure_header(simulate_failure_header)

        cache_key = litellm.cache.get_cache_key(model=self.model, messages=messages)

        # 1. Response Caching Check
        if not sim_map:
            cached_entry = litellm.cache.get_cache(
                model=self.model,
                messages=messages,
                cache={"use-cache": True}
            )
            if cached_entry is None and hasattr(litellm.cache, "cache"):
                raw = litellm.cache.cache.get_cache(cache_key)
                if raw and isinstance(raw, dict) and "response" in raw:
                    cached_entry = raw["response"]

            if cached_entry:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                latency_ms = max(1, latency_ms)

                reply_text = cached_entry.get("reply", "")
                input_tokens = cached_entry.get("input_tokens", 0)
                output_tokens = cached_entry.get("output_tokens", 0)
                cached_model = cached_entry.get("model", self.model)

                return {
                    "reply": reply_text,
                    "resolution_source": "LLM_CACHE",
                    "model": cached_model,
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "key_id": None,
                    "fallback_attempts": 0
                }

        # 2. Key Pool Iteration & Automatic Failover
        key_pool = settings.key_pool
        if not key_pool:
            logger.error("No API keys found in configured key pool")
            raise HTTPException(status_code=503, detail="all providers unavailable")

        fallback_attempts = 0
        last_error = None

        for key_obj in key_pool:
            key_id = key_obj["id"]
            actual_key = key_obj["key"]

            # Check if simulation forces this key to fail
            simulated_code = None
            if "*" in sim_map:
                simulated_code = sim_map["*"]
            elif key_id in sim_map:
                simulated_code = sim_map[key_id]

            if simulated_code is not None:
                # Simulated failure: DO NOT RETRY. Immediately treat as failed and fail over!
                logger.info("Simulating failure on %s with status %s (no retry)", key_id, simulated_code)
                fallback_attempts += 1
                last_error = f"Simulated error {simulated_code}"
                continue

            logger.info("Trying provider %s", key_id)
            key_succeeded = False

            # Real provider call with bounded retries for temporary errors (503, 429, timeout)
            for attempt in range(MAX_ATTEMPTS_PER_KEY):
                try:
                    call_kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "api_key": actual_key,
                        "timeout": self.timeout,
                    }
                    if "gemini-3.8-flash" in self.model:
                        call_kwargs["reasoning_effort"] = "low"

                    # Run blocking litellm.completion off the asyncio event loop
                    response = await asyncio.to_thread(litellm.completion, **call_kwargs)

                    # Extract reply text
                    reply_text = response.choices[0].message.content or ""

                    # Extract token usage
                    if hasattr(response, "usage") and response.usage:
                        input_tokens = getattr(response.usage, "prompt_tokens", None)
                        output_tokens = getattr(response.usage, "completion_tokens", None)
                        if input_tokens is None or output_tokens is None:
                            calc_in, calc_out = calculate_tokens(self.model, messages, reply_text)
                            input_tokens = input_tokens if input_tokens is not None else calc_in
                            output_tokens = output_tokens if output_tokens is not None else calc_out
                    else:
                        input_tokens, output_tokens = calculate_tokens(self.model, messages, reply_text)

                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    latency_ms = max(1, latency_ms)

                    # Store into LiteLLM cache for future identical requests
                    cache_payload = {
                        "reply": reply_text,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "model": self.model
                    }
                    if hasattr(litellm.cache, "cache"):
                        litellm.cache.cache.set_cache(
                            cache_key,
                            {"timestamp": time.time(), "response": cache_payload}
                        )

                    key_succeeded = True
                    return {
                        "reply": reply_text,
                        "resolution_source": "LIVE_INFERENCE",
                        "model": self.model,
                        "latency_ms": latency_ms,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "key_id": key_id,
                        "fallback_attempts": fallback_attempts
                    }

                except Exception as e:
                    last_error = str(e)

                    # 1. Permanent error (401, 403, 404, invalid auth, bad request) -> fail over immediately
                    if is_permanent_error(e):
                        logger.warning(
                            "Permanent provider error on %s: %s. Failing over immediately without retry.",
                            key_id, str(e)
                        )
                        break

                    # 2. Daily/project quota exhaustion (RESOURCE_EXHAUSTED / FreeTier limit) -> skip retries on this key
                    if is_gemini_quota_exhausted(e):
                        logger.warning(
                            "Gemini quota exhausted on %s: %s. Skipping retries on this key.",
                            key_id, str(e)
                        )
                        break

                    # 3. Temporary retryable error (503 high demand, transient 429, timeout, 502/504) -> retry with backoff + jitter
                    if is_retryable_error(e):
                        if attempt < len(BACKOFF_DELAYS):
                            base_delay = BACKOFF_DELAYS[attempt]
                            jitter = random.uniform(0.1, 0.4)
                            delay = base_delay + jitter
                            logger.warning(
                                "Temporary provider error on %s (attempt %d/%d): %s. Retrying in %.2fs...",
                                key_id, attempt + 1, MAX_ATTEMPTS_PER_KEY, str(e), delay
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.warning(
                                "Key %s exhausted all %d retry attempts for temporary error. Failing over to next key.",
                                key_id, MAX_ATTEMPTS_PER_KEY
                            )
                            break
                    else:
                        # Non-retryable provider failure
                        logger.warning(
                            "Non-retryable provider error on %s: %s. Failing over immediately.",
                            key_id, str(e)
                        )
                        break

            if not key_succeeded:
                fallback_attempts += 1

        # If all Gemini providers failed
        logger.error("All providers failed after %s fallback attempts. Last error: %s", fallback_attempts, last_error)
        raise HTTPException(status_code=503, detail="all providers unavailable")


llm_service = LLMService()
