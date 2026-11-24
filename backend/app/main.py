import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import (
    init_db,
    is_db_reachable,
    session_exists,
    get_session_messages,
    save_message,
    get_session_system_prompt,
    save_session_system_prompt,
)
from app.llm_service import llm_service, is_cache_configured
from app.schemas import (
    HealthResponse,
    ChatRequest,
    ChatResponse,
    ChatMetadata,
    SessionMessagesResponse,
    MessageHistoryItem,
    AssistantHistoryMetadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automatically initialize SQLite schema on startup
    init_db()
    logger.info("Database initialized. Ready to accept requests.")
    yield


app = FastAPI(
    title="LLM Chatbot with Key Failover and Response Caching",
    description="FastAPI backend with LiteLLM key failover and response caching",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats validation errors into the exact contract: {'detail': ...} without stack traces."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": jsonable_encoder(exc.errors())}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensures every HTTP error strictly returns JSON with 'detail'."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Never returns stack traces, raw exceptions or provider secrets."""
    logger.error("Unhandled server exception: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint"
)
async def health():
    """
    Returns HTTP 200 only when:
    - Application is running
    - SQLite is reachable
    - Key pool is loaded (>= 2 keys)
    - Cache is configured
    """
    db_ok = is_db_reachable()
    keys_count = llm_service.get_configured_keys_count()
    cache_ok = is_cache_configured()

    if not db_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database is unreachable"
        )

    if keys_count < 2:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="insufficient keys configured in key pool (minimum 2 required)"
        )

    return HealthResponse(
        status="ok",
        keys_configured=keys_count,
        cache_enabled=cache_ok
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message with key failover and caching"
)
async def chat(
    payload: ChatRequest,
    x_simulate_failure: Optional[str] = Header(None, alias="X-Simulate-Failure")
):
    """
    Chat endpoint:
    - Fetches history for session_id from SQLite
    - Injects system prompt
    - Executes via LiteLLM with cache check & failover across key pool
    - Persists user and assistant messages to SQLite upon success
    - Returns exact metadata: resolution_source, model, latency_ms, input_tokens, output_tokens, key_id, fallback_attempts
    """
    session_id = payload.session_id
    user_msg = payload.message

    # 1. System prompt handling
    if payload.system_prompt is not None and payload.system_prompt.strip():
        system_prompt = payload.system_prompt.strip()
        save_session_system_prompt(session_id, system_prompt)
    else:
        existing_prompt = get_session_system_prompt(session_id)
        system_prompt = existing_prompt if existing_prompt else settings.DEFAULT_SYSTEM_PROMPT

    # 2. Retrieve history (oldest first, without system prompt)
    history = get_session_messages(session_id)

    # 3. Build messages list for LiteLLM
    # Cache identity & context: system prompt + history + new user message
    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        llm_messages.append({"role": msg["role"], "content": msg["content"]})
    llm_messages.append({"role": "user", "content": user_msg})

    # 4. Execute inference with caching & automatic key failover
    # If 503 is raised, nothing is inserted into history
    llm_result = await llm_service.execute_chat(
        messages=llm_messages,
        simulate_failure_header=x_simulate_failure
    )

    # 5. On successful completion, persist messages to SQLite
    save_message(
        session_id=session_id,
        role="user",
        content=user_msg,
        metadata=None
    )

    assistant_meta = {
        "resolution_source": llm_result["resolution_source"],
        "input_tokens": llm_result["input_tokens"],
        "output_tokens": llm_result["output_tokens"]
    }
    save_message(
        session_id=session_id,
        role="assistant",
        content=llm_result["reply"],
        metadata=assistant_meta
    )

    # 6. Return response matching exact contract
    return ChatResponse(
        session_id=session_id,
        reply=llm_result["reply"],
        metadata=ChatMetadata(
            resolution_source=llm_result["resolution_source"],
            model=llm_result["model"],
            latency_ms=llm_result["latency_ms"],
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
            key_id=llm_result["key_id"],
            fallback_attempts=llm_result["fallback_attempts"]
        )
    )


@app.get(
    "/sessions/{session_id}/messages",
    response_model=SessionMessagesResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Get conversation history for a session"
)
async def get_messages(session_id: str):
    """
    Returns stored history of one session from SQLite, oldest first, without the system prompt.
    Returns 404 if session is unknown / has no stored messages.
    """
    if not session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="session not found"
        )

    db_messages = get_session_messages(session_id)
    items = []
    for msg in db_messages:
        meta_obj = None
        if msg["metadata"] and msg["role"] == "assistant":
            meta_obj = AssistantHistoryMetadata(
                resolution_source=msg["metadata"].get("resolution_source", "LIVE_INFERENCE"),
                input_tokens=msg["metadata"].get("input_tokens", 0),
                output_tokens=msg["metadata"].get("output_tokens", 0)
            )

        items.append(MessageHistoryItem(
            role=msg["role"],
            content=msg["content"],
            metadata=meta_obj
        ))

    return SessionMessagesResponse(
        session_id=session_id,
        messages=items
    )
