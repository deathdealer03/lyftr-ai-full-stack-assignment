from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    keys_configured: int
    cache_enabled: bool


class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("session_id must be a string")
        v = v.strip()
        if not (1 <= len(v) <= 128):
            raise ValueError("session_id must be between 1 and 128 characters")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("message must be a string")
        v = v.strip()
        if not (1 <= len(v) <= 4000):
            raise ValueError("message must be between 1 and 4000 characters after trimming whitespace")
        return v


class ChatMetadata(BaseModel):
    resolution_source: str = Field(..., description="LIVE_INFERENCE or LLM_CACHE")
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str] = None
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class AssistantHistoryMetadata(BaseModel):
    resolution_source: str
    input_tokens: int
    output_tokens: int


class MessageHistoryItem(BaseModel):
    role: str
    content: str
    metadata: Optional[AssistantHistoryMetadata] = None


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: List[MessageHistoryItem]


class ErrorDetail(BaseModel):
    detail: Any
