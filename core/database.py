"""
Async PostgreSQL connection pool using asyncpg.
pgvector codec is registered on every new connection so vector columns
can be read/written as native Python lists without any manual serialization.

Usage (from FastAPI lifespan or anywhere async):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT ...")
"""
from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from config import get_settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Called once per connection on creation — registers the vector codec."""
    await register_vector(conn)


async def get_pool() -> asyncpg.Pool:
    """
    Returns (or lazily creates) the shared async connection pool.
    Pool size: 2 min / 10 max — tuned for a single FastAPI instance.
    Raise the max_size to ~20 in production k8s horizontal pod autoscaling.
    """
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            init=_init_connection,
        )
    return _pool


async def close_pool() -> None:
    """Gracefully close the pool — call from FastAPI shutdown lifespan event."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
