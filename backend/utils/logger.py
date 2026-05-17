# ============================================================
# File: backend/utils/logger.py
# Purpose: Centralized structured logging using loguru.
#          Configures console + file output, rotation, and
#          environment-aware formatting (pretty vs JSON).
#
# Used by: Every backend file that needs logging:
#   - backend/db/database.py
#   - backend/services/*.py
#   - backend/agents/*.py
#   - backend/orchestrator/*.py
#   - backend/api/routes/*.py
#   - backend/main.py
#
# Usage:
#   from backend.utils.logger import logger
#   logger.info("Something happened")
#   logger.error("Something broke: {error}", error=str(e))
# ============================================================

import json
import logging
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger

from backend.config import settings


# ============================================================
# Loguru Configuration
# ============================================================

def _build_log_format(is_production: bool) -> str:
    """
    Returns the log format string based on environment.

    Development: Colored, human-readable with emoji level indicators.
    Production:  Plain structured format (JSON sink handles serialization).

    Args:
        is_production: True when APP_ENV=production.

    Returns:
        str: Loguru format string.
    """
    if is_production:
        # Clean format — JSON sink will serialize it
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

    # Development: colorized and easy to read in terminal
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )


def _json_sink(message: Any) -> None:
    """
    Custom JSON sink for production structured logging.

    Serializes each log record as a single-line JSON object.
    This format is compatible with Datadog, CloudWatch,
    Elasticsearch, and most log aggregation platforms.

    Args:
        message: Loguru message record object.
    """
    record = message.record

    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
    }

    # Include exception info if present
    if record["exception"]:
        exc = record["exception"]
        log_entry["exception"] = {
            "type": exc.type.__name__ if exc.type else None,
            "value": str(exc.value) if exc.value else None,
        }

    # Include any extra fields passed via logger.bind() or logger.info(..., key=val)
    if record["extra"]:
        log_entry["extra"] = record["extra"]

    # Write JSON line to stderr (stdout is reserved for app output)
    print(json.dumps(log_entry), file=sys.stderr)


def setup_logger() -> None:
    """
    Configures the global loguru logger.

    Called once at module import time.
    Sets up:
        1. Console sink  — always active, format depends on environment
        2. File sink     — active when LOG_OUTPUT includes 'file'
        3. JSON sink     — active in production for structured logging

    Log levels map:
        DEBUG    → Detailed internal state (dev only)
        INFO     → Normal operational events
        WARNING  → Unexpected but recoverable situations
        ERROR    → Failures that need attention
        CRITICAL → System-level failures, app may not continue
    """

    # Remove loguru's default handler before adding our own
    _loguru_logger.remove()

    log_level = settings.log_level      # Already validated + uppercased in config.py
    log_output = settings.log_output.lower()

    # ----------------------------------------------------------
    # Sink 1: Console Output
    # ----------------------------------------------------------
    # Always active — shows logs in the terminal.
    # In production, format is plain. In dev, colorized.

    if log_output in ("console", "both"):
        _loguru_logger.add(
            sink=sys.stderr,
            level=log_level,
            format=_build_log_format(settings.is_production),
            colorize=not settings.is_production,
            backtrace=settings.debug,           # Show full traceback in debug mode
            diagnose=settings.debug,            # Show variable values in tracebacks
            enqueue=False,                      # Synchronous for console
        )

    # ----------------------------------------------------------
    # Sink 2: File Output with Rotation
    # ----------------------------------------------------------
    # Writes logs to a file with automatic rotation and retention.
    # Useful for debugging issues after the fact.

    if log_output in ("file", "both"):
        log_path = settings.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        _loguru_logger.add(
            sink=str(log_path),
            level=log_level,
            format=_build_log_format(is_production=False),  # Always readable in file
            rotation=f"{settings.log_max_size_mb} MB",      # Rotate when file reaches limit
            retention=settings.log_backup_count,             # Keep N rotated files
            compression="zip",                               # Compress old log files
            backtrace=True,
            diagnose=settings.debug,
            enqueue=True,                                    # Async writes — no I/O blocking
            encoding="utf-8",
        )

    # ----------------------------------------------------------
    # Sink 3: JSON Structured Output (Production Only)
    # ----------------------------------------------------------
    # Outputs one JSON object per line to stderr.
    # Designed for ingestion by log aggregation platforms.

    if settings.is_production:
        _loguru_logger.add(
            sink=_json_sink,
            level="INFO",          # Only INFO+ in production JSON logs
            format="{message}",    # Raw message — _json_sink handles formatting
            serialize=False,       # We handle serialization in _json_sink
            backtrace=False,       # No backtraces in JSON production logs
            diagnose=False,
        )

    _loguru_logger.info(
        "Logger initialized | "
        f"level={log_level} | "
        f"output={log_output} | "
        f"env={settings.app_env}"
    )


# ============================================================
# Intercept Standard Library Logging
# ============================================================
# Many libraries (SQLAlchemy, FastAPI, Uvicorn, Celery) use
# Python's built-in `logging` module. This intercepts those
# messages and routes them through loguru so everything
# appears in the same format and goes to the same files.

class _InterceptHandler(logging.Handler):
    """
    Routes standard library logging records to loguru.

    Install with:
        logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Map standard logging level to loguru level name
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find the correct call stack depth so loguru reports
        # the ORIGINAL caller location, not this handler
        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _intercept_standard_logging() -> None:
    """
    Replaces all standard library log handlers with the loguru interceptor.

    This captures log output from:
        - SQLAlchemy      (database queries)
        - Uvicorn         (server access logs)
        - FastAPI         (request/response events)
        - Celery          (task lifecycle events)
        - httpx           (outgoing HTTP requests)
        - asyncio         (event loop warnings)
    """
    # Libraries whose logs we want to capture
    intercept_libraries = [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "celery",
        "celery.task",
        "httpx",
        "asyncio",
        "playwright",
        "aiohttp",
    ]

    logging.basicConfig(
        handlers=[_InterceptHandler()],
        level=0,            # Capture ALL levels — loguru filters by its own level
        force=True,         # Override any existing handlers
    )

    for lib_name in intercept_libraries:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.handlers = [_InterceptHandler()]
        lib_logger.propagate = False    # Don't pass to root logger again


# ============================================================
# Agent-Specific Logger Factory
# ============================================================

def get_agent_logger(agent_name: str):
    """
    Returns a loguru logger instance bound to a specific agent.

    All log messages from this logger automatically include
    the agent name as a structured field — making it easy to
    filter logs for a specific agent in production.

    Args:
        agent_name: Human-readable name of the agent
                    e.g. "ResumeParserAgent", "JobMatchingAgent"

    Returns:
        A loguru logger with agent context bound.

    Usage in agent files:
        from backend.utils.logger import get_agent_logger

        class ResumeParserAgent:
            def __init__(self):
                self.logger = get_agent_logger("ResumeParserAgent")

            async def run(self):
                self.logger.info("Starting resume parsing...")
                # Logs: "Starting resume parsing..." | agent=ResumeParserAgent
    """
    return _loguru_logger.bind(agent=agent_name)


def get_service_logger(service_name: str):
    """
    Returns a loguru logger bound to a specific service.

    Usage in service files:
        from backend.utils.logger import get_service_logger

        class LLMService:
            def __init__(self):
                self.logger = get_service_logger("LLMService")
    """
    return _loguru_logger.bind(service=service_name)


# ============================================================
# Pipeline Step Logger
# ============================================================

def log_pipeline_step(
    step: int,
    total: int,
    agent_name: str,
    status: str,
    detail: str = "",
) -> None:
    """
    Logs a pipeline step with consistent formatting.

    Called by the orchestrator as each agent starts/completes.
    Makes pipeline progress easy to trace in logs.

    Args:
        step:       Current step number (1-based)
        total:      Total number of steps in pipeline
        agent_name: Name of the agent running this step
        status:     "started" | "completed" | "failed" | "skipped"
        detail:     Optional additional context

    Output example:
        [Pipeline 3/8] JobMatchingAgent → completed | match_count=15
    """
    status_symbol = {
        "started":   "▶",
        "completed": "✓",
        "failed":    "✗",
        "skipped":   "⊘",
    }.get(status, "•")

    message = (
        f"[Pipeline {step}/{total}] "
        f"{status_symbol} {agent_name} → {status}"
    )

    if detail:
        message += f" | {detail}"

    if status == "failed":
        _loguru_logger.error(message)
    elif status == "completed":
        _loguru_logger.success(message)
    elif status == "skipped":
        _loguru_logger.warning(message)
    else:
        _loguru_logger.info(message)


# ============================================================
# Module Initialization
# ============================================================
# These run once when this module is first imported.

setup_logger()
_intercept_standard_logging()


# ============================================================
# Public Export
# ============================================================
# Other files import ONLY `logger` from this module.
# The underlying loguru instance is re-exported with a clean name.

logger = _loguru_logger