# ============================================================
# File: backend/db/database.py
# Purpose: Async SQLAlchemy engine, session factory, and Base
#          model class for the entire application.
#
# Used by:
#   - backend/models/*.py          → inherit from Base
#   - backend/db/repositories/*.py → use AsyncSession
#   - backend/api/routes/*.py      → use get_db() dependency
#   - backend/main.py              → call init_db() on startup
# ============================================================

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, MappedColumn
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from backend.config import settings
from backend.utils.logger import logger


# ============================================================
# Declarative Base
# ============================================================
# All database models (Resume, Job, Application, User, etc.)
# inherit from this Base class. SQLAlchemy uses it to discover
# all tables when running migrations or creating the schema.

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Every model file does:
        from backend.db.database import Base
        class MyModel(Base):
            __tablename__ = "my_table"
            ...
    """
    pass


# ============================================================
# Engine Factory
# ============================================================

def _create_engine() -> AsyncEngine:
    """
    Creates and configures the async SQLAlchemy engine.

    - Development: Uses connection pooling for performance.
    - Testing:     Uses NullPool (no pooling) for clean test isolation.
    - Production:  Uses larger pool with pre-ping for reliability.

    Returns:
        AsyncEngine: The configured async database engine.
    """
    # Choose pool class based on environment
    # NullPool creates a new connection per request — good for tests
    # AsyncAdaptedQueuePool reuses connections — good for production
    pool_class = NullPool if settings.app_env == "testing" else AsyncAdaptedQueuePool

    engine = create_async_engine(
        url=settings.database_url,

        # Echo SQL queries to console in debug mode only
        echo=settings.debug and settings.is_development,

        # Pool configuration (ignored when NullPool is used)
        poolclass=pool_class,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,

        # Pre-ping checks the connection is alive before using it.
        # Prevents "connection closed" errors after DB restarts.
        pool_pre_ping=True,

        # How long to wait for a connection from the pool (seconds)
        pool_timeout=30,

        # How long a connection can sit idle before being recycled
        pool_recycle=1800,   # 30 minutes

        # JSON serializer — use orjson for speed if available
        json_serializer=_json_serializer,
        json_deserializer=_json_deserializer,
    )

    logger.info(
        f"Database engine created | "
        f"host={settings.db_host} | "
        f"db={settings.db_name} | "
        f"pool_size={settings.db_pool_size}"
    )

    return engine


def _json_serializer(obj) -> str:
    """
    Fast JSON serializer for SQLAlchemy JSON columns.
    Falls back to standard json if orjson is unavailable.
    """
    try:
        import orjson
        return orjson.dumps(obj).decode("utf-8")
    except ImportError:
        import json
        return json.dumps(obj)


def _json_deserializer(data: str) -> dict:
    """
    Fast JSON deserializer for SQLAlchemy JSON columns.
    Falls back to standard json if orjson is unavailable.
    """
    try:
        import orjson
        return orjson.loads(data)
    except ImportError:
        import json
        return json.loads(data)


# ============================================================
# Engine & Session Factory — Module-Level Singletons
# ============================================================

# The engine is created once when this module is first imported.
# It manages the connection pool for the entire application lifetime.
engine: AsyncEngine = _create_engine()

# Session factory — call AsyncSessionLocal() to get a new session.
# expire_on_commit=False means objects stay usable after commit,
# which is important in async code where you access attrs after await.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ============================================================
# FastAPI Dependency — get_db()
# ============================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.

    Opens a session, yields it to the route handler, then always
    closes it — even if an exception is raised.

    Usage in any API route:
        from fastapi import Depends
        from backend.db.database import get_db
        from sqlalchemy.ext.asyncio import AsyncSession

        @router.get("/jobs")
        async def get_jobs(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Job))
            return result.scalars().all()

    The session is automatically:
        - Committed on success
        - Rolled back on any exception
        - Closed after the response is sent
    """
    async with AsyncSessionLocal() as session:
        async with session.begin():
            try:
                yield session
            except Exception as exc:
                await session.rollback()
                logger.error(f"Database session error: {exc}")
                raise
            finally:
                await session.close()


# ============================================================
# Context Manager — for use outside FastAPI routes
# ============================================================

@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions outside of
    FastAPI route handlers (e.g. in agents, services, scripts).

    Usage:
        from backend.db.database import get_db_context

        async with get_db_context() as db:
            result = await db.execute(select(Resume))
            resumes = result.scalars().all()

    This is what agents and background Celery tasks use
    since they don't have access to FastAPI's Depends system.
    """
    async with AsyncSessionLocal() as session:
        async with session.begin():
            try:
                yield session
            except Exception as exc:
                await session.rollback()
                logger.error(f"Database context error: {exc}")
                raise
            finally:
                await session.close()


# ============================================================
# Database Initialization
# ============================================================

async def init_db() -> None:
    """
    Creates all database tables on application startup.

    Called from backend/main.py lifespan event.

    IMPORTANT: This uses SQLAlchemy's create_all which is fine
    for development. In production, use Alembic migrations instead:
        alembic upgrade head

    This function:
        1. Imports all models so Base knows about their tables
        2. Creates tables that don't exist yet
        3. Skips tables that already exist (safe to call repeatedly)
    """
    # Import all models here so Base.metadata knows about them.
    # This is the ONLY place we do wildcard-style model imports.
    # If you add a new model, register it here.
    from backend.models import (  # noqa: F401 — imported for side effects
        user,
        resume,
        job,
        application,
    )

    async with engine.begin() as conn:
        logger.info("Initializing database — creating tables if needed...")

        # create_all is idempotent — safe to call on every startup
        await conn.run_sync(Base.metadata.create_all)

        logger.info(
            f"Database initialized | "
            f"tables={list(Base.metadata.tables.keys())}"
        )


async def drop_db() -> None:
    """
    Drops ALL tables. Used in tests only — never in production.

    Usage in test fixtures:
        await drop_db()
        await init_db()
    """
    if settings.is_production:
        raise RuntimeError(
            "drop_db() is forbidden in production. "
            "Use Alembic migrations to manage schema changes."
        )

    async with engine.begin() as conn:
        logger.warning("Dropping all database tables...")
        await conn.run_sync(Base.metadata.drop_all)
        logger.warning("All tables dropped.")


# ============================================================
# Health Check
# ============================================================

async def check_db_connection() -> dict:
    """
    Verifies the database is reachable and responsive.

    Called from:
        - backend/main.py startup validation
        - GET /health API endpoint

    Returns:
        dict: Status info including latency and PostgreSQL version.

    Example return:
        {
            "status": "healthy",
            "latency_ms": 2.4,
            "version": "PostgreSQL 15.3"
        }
    """
    import time

    try:
        start = time.perf_counter()

        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT version()"))
            version_row = result.fetchone()
            version = version_row[0] if version_row else "unknown"

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.debug(f"DB health check passed | latency={latency_ms}ms")

        return {
            "status": "healthy",
            "latency_ms": latency_ms,
            "version": version.split(",")[0],   # Clean up the version string
        }

    except Exception as exc:
        logger.error(f"DB health check failed: {exc}")
        return {
            "status": "unhealthy",
            "error": str(exc),
        }


# ============================================================
# Graceful Shutdown
# ============================================================

async def close_db() -> None:
    """
    Disposes the connection pool on application shutdown.

    Called from backend/main.py lifespan shutdown event.
    Ensures all active connections are cleanly closed before
    the process exits — prevents "connection leak" warnings.
    """
    logger.info("Closing database connection pool...")
    await engine.dispose()
    logger.info("Database connection pool closed.")