from __future__ import annotations

import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://aether:aether@localhost:5432/aether_eye",
)

SQL_ECHO = os.getenv("SQL_ECHO", "false").lower() == "true"


class Base(DeclarativeBase):
    pass


engine = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _ensure_engine_and_session() -> tuple[object, async_sessionmaker[AsyncSession]]:
    global engine, _async_session_factory
    if engine is None or _async_session_factory is None:
        engine = create_async_engine(
            DATABASE_URL,
            echo=SQL_ECHO,
            pool_pre_ping=True,
        )
        _async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return engine, _async_session_factory


def async_session() -> AsyncSession:
    _, factory = _ensure_engine_and_session()
    return factory()


async def get_db():
    async with async_session() as session:
        yield session


async def init_db():
    db_engine, _ = _ensure_engine_and_session()
    if not DATABASE_URL.startswith("postgresql"):
        return

    async with db_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
