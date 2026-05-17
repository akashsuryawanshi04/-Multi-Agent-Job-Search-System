# ============================================================
# File: backend/config.py
# Purpose: Centralized configuration using pydantic-settings
#          Reads from .env, validates types, exposes `settings`
# Used by: Every backend file that needs config or secrets
# ============================================================

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import AnyHttpUrl, EmailStr, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# Base directory of the project (job-search-ai/)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Main Settings Class
# ============================================================

class Settings(BaseSettings):
    """
    Central settings class for the Job Search AI backend.

    All values are loaded from the .env file in the project root.
    Pydantic validates types automatically — wrong types cause
    a startup error with a clear message.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",          # Path to .env file
        env_file_encoding="utf-8",
        case_sensitive=False,                # ENV_VAR == env_var
        extra="ignore",                      # Ignore unknown vars in .env
    )

    # ----------------------------------------------------------
    # Application Settings
    # ----------------------------------------------------------
    app_env: str = Field(default="development", description="Environment name")
    app_name: str = Field(default="Job Search AI")
    app_version: str = Field(default="1.0.0")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    debug: bool = Field(default=True)

    # Comma-separated string from .env → parsed into a Python list
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000"
    )

    @property
    def allowed_origins_list(self) -> List[str]:
        """Returns allowed origins as a clean Python list."""
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """True when running in development environment."""
        return self.app_env.lower() == "development"

    # ----------------------------------------------------------
    # Anthropic (Claude AI)
    # ----------------------------------------------------------
    anthropic_api_key: str = Field(
        ...,                                 # Required — no default
        description="Anthropic API key from console.anthropic.com"
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-5",
        description="Claude model to use for all agents"
    )
    anthropic_max_tokens: int = Field(
        default=4096,
        description="Max tokens returned per Claude API call"
    )
    anthropic_timeout: int = Field(
        default=60,
        description="Timeout in seconds for Claude API calls"
    )
    anthropic_max_retries: int = Field(
        default=3,
        description="Max retry attempts on failed Claude API calls"
    )

    # ----------------------------------------------------------
    # Database (PostgreSQL)
    # ----------------------------------------------------------
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="jobsearch_db")
    db_user: str = Field(default="postgres")
    db_password: str = Field(..., description="PostgreSQL password")
    db_pool_size: int = Field(default=10)
    db_max_overflow: int = Field(default=20)

    @property
    def database_url(self) -> str:
        """
        Async PostgreSQL URL used by SQLAlchemy engine.
        Format: postgresql+asyncpg://user:pass@host:port/dbname
        """
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def sync_database_url(self) -> str:
        """
        Sync PostgreSQL URL used by Alembic migrations only.
        Format: postgresql+psycopg2://user:pass@host:port/dbname
        """
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ----------------------------------------------------------
    # Redis
    # ----------------------------------------------------------
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str = Field(default="")
    redis_db: int = Field(default=0)

    @property
    def redis_url(self) -> str:
        """
        Full Redis connection URL.
        Includes password if set, omits it if empty.
        """
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}"
                f"@{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ----------------------------------------------------------
    # Celery (Task Queue)
    # ----------------------------------------------------------
    celery_broker_url: str = Field(default="redis://localhost:6379/0")
    celery_result_backend: str = Field(default="redis://localhost:6379/1")
    celery_workers: int = Field(default=4)
    celery_task_timeout: int = Field(default=300)

    # ----------------------------------------------------------
    # JWT Authentication
    # ----------------------------------------------------------
    jwt_secret_key: str = Field(
        ...,
        description="Secret key for signing JWT tokens. "
                    "Generate with: openssl rand -hex 32"
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=60)
    jwt_refresh_token_expire_days: int = Field(default=7)

    # ----------------------------------------------------------
    # File Uploads
    # ----------------------------------------------------------
    upload_dir: str = Field(default="uploads")
    max_upload_size_mb: int = Field(default=10)
    allowed_file_types: str = Field(default="pdf,docx,doc")

    @property
    def upload_path(self) -> Path:
        """Absolute path to the uploads directory."""
        path = BASE_DIR / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)   # Create if not exists
        return path

    @property
    def max_upload_size_bytes(self) -> int:
        """Max upload size converted to bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def allowed_file_types_list(self) -> List[str]:
        """Returns allowed file types as a Python list."""
        return [ft.strip().lower() for ft in self.allowed_file_types.split(",")]

    # ----------------------------------------------------------
    # Job Scraping (RapidAPI / JSearch)
    # ----------------------------------------------------------
    rapidapi_key: str = Field(
        default="",
        description="RapidAPI key for JSearch job search API"
    )
    rapidapi_host: str = Field(default="jsearch.p.rapidapi.com")
    jsearch_base_url: str = Field(default="https://jsearch.p.rapidapi.com")
    job_search_max_results: int = Field(default=50)
    scraper_timeout: int = Field(default=30)
    scraper_delay: float = Field(default=2.0)

    # ----------------------------------------------------------
    # Email Service
    # ----------------------------------------------------------
    mail_username: str = Field(default="")
    mail_password: str = Field(default="")
    mail_from: str = Field(default="noreply@jobsearchai.com")
    mail_from_name: str = Field(default="Job Search AI")
    mail_port: int = Field(default=587)
    mail_server: str = Field(default="smtp.gmail.com")
    mail_starttls: bool = Field(default=True)
    mail_ssl_tls: bool = Field(default=False)
    mail_enabled: bool = Field(default=False)

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------
    log_level: str = Field(default="DEBUG")
    log_output: str = Field(default="both")
    log_file_path: str = Field(default="logs/app.log")
    log_max_size_mb: int = Field(default=10)
    log_backup_count: int = Field(default=5)

    @property
    def log_path(self) -> Path:
        """Absolute path to the log file."""
        path = BASE_DIR / self.log_file_path
        path.parent.mkdir(parents=True, exist_ok=True)   # Create logs/ dir
        return path

    # ----------------------------------------------------------
    # NLP Models
    # ----------------------------------------------------------
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for job matching"
    )
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="SpaCy model for NER in resume parsing"
    )
    min_match_score: float = Field(
        default=0.4,
        description="Minimum cosine similarity to include a job match"
    )

    # ----------------------------------------------------------
    # Pipeline Settings
    # ----------------------------------------------------------
    pipeline_max_jobs: int = Field(
        default=20,
        description="Max jobs to process per pipeline run"
    )
    pipeline_parallel_enabled: bool = Field(
        default=True,
        description="Run ATS + Cover Letter agents concurrently"
    )
    ws_ping_interval: int = Field(default=20)
    ws_ping_timeout: int = Field(default=10)

    # ----------------------------------------------------------
    # Validators
    # ----------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensures log level is a valid Python logging level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(
                f"Invalid LOG_LEVEL '{v}'. Must be one of: {valid}"
            )
        return upper

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Ensures app environment is a recognized value."""
        valid = {"development", "staging", "production"}
        lower = v.lower()
        if lower not in valid:
            raise ValueError(
                f"Invalid APP_ENV '{v}'. Must be one of: {valid}"
            )
        return lower

    @field_validator("min_match_score")
    @classmethod
    def validate_min_match_score(cls, v: float) -> float:
        """Ensures match score threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"MIN_MATCH_SCORE must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("anthropic_model")
    @classmethod
    def validate_anthropic_model(cls, v: str) -> str:
        """Ensures the configured model is a known Claude model."""
        valid_prefixes = ("claude-opus", "claude-sonnet", "claude-haiku")
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid ANTHROPIC_MODEL '{v}'. "
                f"Must start with one of: {valid_prefixes}"
            )
        return v

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """
        Extra validation rules that only apply in production.
        Prevents insecure configs from reaching live servers.
        """
        if self.is_production:
            if self.debug:
                raise ValueError(
                    "DEBUG must be False in production environment."
                )
            if self.jwt_secret_key == "your-super-secret-jwt-key-generate-with-openssl":
                raise ValueError(
                    "You must set a real JWT_SECRET_KEY in production."
                )
            if len(self.jwt_secret_key) < 32:
                raise ValueError(
                    "JWT_SECRET_KEY must be at least 32 characters in production."
                )
        return self


# ============================================================
# Singleton Pattern using lru_cache
# ============================================================
# @lru_cache ensures Settings() is only instantiated ONCE.
# Every file that calls get_settings() gets the same object.
# This avoids re-reading .env on every import.

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached singleton Settings instance.

    Usage in any file:
        from backend.config import get_settings
        settings = get_settings()

    Usage as FastAPI dependency:
        from fastapi import Depends
        def my_route(settings: Settings = Depends(get_settings)):
            ...
    """
    return Settings()


# ============================================================
# Module-level singleton for direct import convenience
# ============================================================
# This lets files do:
#   from backend.config import settings
# instead of:
#   from backend.config import get_settings
#   settings = get_settings()

settings: Settings = get_settings()


# ============================================================
# Startup validation — called once when the app boots
# ============================================================

def validate_critical_settings() -> None:
    """
    Checks that all critical external services are configured.
    Called from main.py on application startup.
    Raises RuntimeError with a clear message if anything is missing.
    """
    errors: List[str] = []

    # Claude API key must look like a real Anthropic key
    if not settings.anthropic_api_key.startswith("sk-ant-"):
        errors.append(
            "ANTHROPIC_API_KEY appears invalid. "
            "Get your key from https://console.anthropic.com/"
        )

    # Database password must be set
    if not settings.db_password:
        errors.append(
            "DB_PASSWORD is not set. "
            "PostgreSQL requires a password."
        )

    # JWT secret must be changed from the placeholder
    if settings.jwt_secret_key == "your-super-secret-jwt-key-generate-with-openssl":
        errors.append(
            "JWT_SECRET_KEY is still the placeholder value. "
            "Run: openssl rand -hex 32"
        )

    if errors:
        error_list = "\n  ".join(f"- {e}" for e in errors)
        raise RuntimeError(
            f"\n\n{'='*60}\n"
            f"STARTUP FAILED — Missing or invalid configuration:\n\n"
            f"  {error_list}\n\n"
            f"Please check your .env file.\n"
            f"{'='*60}\n"
        )