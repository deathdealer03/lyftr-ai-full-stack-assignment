import os
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
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated allowed CORS origins"
    )
    ENABLE_FAILURE_SIMULATION: bool = Field(
        default=True,
        description="Enable X-Simulate-Failure header"
    )
    LLM_MODEL: str = Field(
        default="gemini/gemini-2.0-flash",
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

    @property
    def key_list(self) -> List[str]:
        keys = [k.strip() for k in self.LLM_API_KEYS.split(",") if k.strip()]
        return keys

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
