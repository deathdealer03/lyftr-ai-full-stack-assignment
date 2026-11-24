import os
import time
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
)

from app.config import settings

logger = logging.getLogger("backend.llm")
litellm.telemetry = False
litellm.suppress_debug_info = True

# Initialize LiteLLM Cache
cache_type = settings.CACHE_TYPE if settings.CACHE_TYPE in ("local", "disk") else "local"
litellm.cache = Cache(type=cache_type, mode="default_on")
litellm.enable_cache(type=cache_type)


def is_cache_configured() -> bool:
    """Returns True if LiteLLM cache is configured."""
    return litellm.cache is not None


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
        # Fallback estimation if model tokenizer unavailable
        total_chars = sum(len(m.get("content", "")) for m in messages)
        input_tokens = max(1, total_chars // 4)

    try:
        output_tokens = litellm.token_counter(model=model, text=reply_text)
    except Exception:
        output_tokens = max(1, len(reply_text) // 4)

    return int(input_tokens), int(output_tokens)


class LLMService:
    def __init__(self):
        self.model = settings.LLM_MODEL

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
        3. X-Simulate-Failure test hook handling
        4. Token tracking (input_tokens for this request, output_tokens for reply)
        5. Accurate latency measurement
        """
        start_time = time.perf_counter()
        sim_map = {}
        if settings.ENABLE_FAILURE_SIMULATION and simulate_failure_header:
            sim_map = parse_simulate_failure_header(simulate_failure_header)

        cache_key = litellm.cache.get_cache_key(model=self.model, messages=messages)

        # 1. Response Caching Check
        # If no simulation requested, check LiteLLM Cache first
        if not sim_map:
            cached_entry = litellm.cache.get_cache(
                model=self.model,
                messages=messages,
                cache={"use-cache": True}
            )
            # If not found via high-level helper, check underlying store
            if cached_entry is None and hasattr(litellm.cache, "cache"):
                raw = litellm.cache.cache.get_cache(cache_key)
                if raw and isinstance(raw, dict) and "response" in raw:
                    cached_entry = raw["response"]

            if cached_entry:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                # Ensure minimum 1ms or realistic cache retrieval latency
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
                logger.info("Simulating failure on %s with status %s", key_id, simulated_code)
                fallback_attempts += 1
                last_error = f"Simulated error {simulated_code}"
                continue

            # Live call using LiteLLM
            try:
                # Direct call to LiteLLM completion
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    api_key=actual_key,
                    timeout=30.0
                )

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

            except (RateLimitError, ServiceUnavailableError, InternalServerError, BadGatewayError, APIConnectionError, APIError) as e:
                logger.warning("Provider error with %s: %s", key_id, str(e))
                fallback_attempts += 1
                last_error = str(e)
                continue
            except Exception as e:
                # Check status code attribute if present on exception
                status_code = getattr(e, "status_code", None)
                if status_code in (429, 500, 502, 503):
                    logger.warning("HTTP status %s with %s", status_code, key_id)
                    fallback_attempts += 1
                    last_error = str(e)
                    continue
                else:
                    # Treat unexpected provider call failure as fallback opportunity
                    logger.warning("Error with %s: %s", key_id, str(e))
                    fallback_attempts += 1
                    last_error = str(e)
                    continue

        # If every key in the pool failed
        logger.error("All providers failed after %s fallback attempts. Last error: %s", fallback_attempts, last_error)
        raise HTTPException(status_code=503, detail="all providers unavailable")


llm_service = LLMService()
