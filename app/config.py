"""
Central configuration. Everything secret/environment-specific comes from
environment variables (.env locally, real env vars / secret manager in
production). Never hardcode API keys, endpoints, or credentials here.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "ITSM Quality Analysis Agent"
    ENV: Literal["dev", "staging", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"

    # --- Security ---
    # Comma-separated list of valid API keys for calling the backend.
    # In prod, back this with a secrets manager / SAP BTP destination service.
    API_KEYS: str = "change-me-local-dev-key"
    ALLOWED_ORIGINS: str = "http://localhost:7860,http://localhost:8000"
    MAX_UPLOAD_MB: int = 15
    RATE_LIMIT: str = "30/minute"

    # --- LLM provider selection ---
    # "sap_genai_hub"  -> uses SAP Generative AI Hub SDK (orchestration/proxy)
    # "openai_compat"  -> any OpenAI-compatible endpoint (incl. internal gateways)
    # "rule_based"     -> no LLM calls at all, pure heuristics (safe default/demo)
    LLM_PROVIDER: Literal["sap_genai_hub", "openai_compat", "rule_based"] = "rule_based"

    # Model identifiers as provisioned in your model catalog / SAP AI Core
    # deployment config. These are names, not endpoints - the SDK/gateway
    # resolves them to actual deployments.
    CHAT_MODEL_NAME: str = "gpt-5.6-luna"
    EMBEDDING_MODEL_NAME: str = "text-embedding-large"

    # SAP AI Core / Generative AI Hub (used only if LLM_PROVIDER=sap_genai_hub)
    AICORE_AUTH_URL: str | None = None
    AICORE_CLIENT_ID: str | None = None
    AICORE_CLIENT_SECRET: str | None = None
    AICORE_BASE_URL: str | None = None
    AICORE_RESOURCE_GROUP: str | None = None

    # Generic OpenAI-compatible gateway (used only if LLM_PROVIDER=openai_compat)
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: str | None = None

    # --- Categorization ---
    SIMILARITY_THRESHOLD: float = 0.35  # min cosine sim to accept embedding match


@lru_cache
def get_settings() -> Settings:
    return Settings()
