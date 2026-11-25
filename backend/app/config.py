import os
import re
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Ensure LiteLLM uses local offline cost map to prevent network hangs on startup
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    LLM_API_KEYS: str = Field(
        default="mock_key_alpha,mock_key_beta",
        description="Comma-separated API keys pool"
    )
    DATABASE_URL: str = Field(
        default="sqlite:///./data/app.db",
        description="SQLite database URL"
    )
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,https://lyftr-ai-full-stack-assignment.vercel.app",
        description="Comma-separated allowed CORS origins"
    )
    ENABLE_FAILURE_SIMULATION: bool = Field(
        default=True,
        description="Enable X-Simulate-Failure header"
    )
    LLM_MODEL: str = Field(
        default="gemini/gemini-3.5-flash-lite",
        description="LiteLLM model name to use"
    )
    DEFAULT_SYSTEM_PROMPT: str = Field(
        default="You are a helpful and concise AI assistant.",
        description="Default system prompt for new sessions"
    )
    CACHE_TYPE: str = Field(
        default="local",
        description="LiteLLM cache type: local or disk"
    )
    LLM_REQUEST_TIMEOUT: float = Field(
        default=20.0,
        description="Per-request timeout in seconds for LiteLLM completion calls"
    )

    @property
    def normalized_model(self) -> str:
        """
        Normalizes model string to ensure Google AI Studio (gemini) provider routing in LiteLLM.
        If a user supplies 'gemini-3.8-flash' without 'gemini/', LiteLLM defaults to vertex_ai.
        Prefixed with 'gemini/', LiteLLM correctly routes to Google AI Studio via Gemini API keys.
        """
        raw = self.LLM_MODEL.strip().strip("'\"")
        if raw.startswith("gemini-") and not raw.startswith("gemini/"):
            return f"gemini/{raw}"
        return raw

    @property
    def key_list(self) -> List[str]:
        raw = self.LLM_API_KEYS
        if not raw:
            return []
        raw = raw.strip()
        # Strip enclosing single/double quotes around entire env value
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1].strip()

        # Normalize line endings and delimiters (commas, newlines, semicolons)
        normalized = raw.replace("\r\n", "\n").replace("\r", "\n").replace(";", ",")
        tokens = re.split(r"[\n,]+", normalized)

        cleaned: List[str] = []
        for token in tokens:
            k = token.strip().strip("'\"").strip()
            if k:
                cleaned.append(k)
        return cleaned

    @property
    def key_pool(self) -> List[dict]:
        """
        Returns list of safe internal keys:
        [{ "id": "key_1", "key": "..." }, { "id": "key_2", "key": "..." }]
        """
        return [{"id": f"key_{i+1}", "key": key} for i, key in enumerate(self.key_list)]

    @property
    def cors_origins(self) -> List[str]:
        origins = [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]
        # If wildcard or specific origins
        return origins if origins else ["*"]


settings = Settings()

