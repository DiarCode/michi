"""Shared Redis connection pool — single source of truth for REDIS_URL."""

import logging

import redis

from backend.config import REDIS_URL

logger = logging.getLogger(__name__)

# Module-level connection pool — shared across all modules that import from here.
# This prevents connection leaks: each process gets one pool with reusable connections.
_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True, max_connections=10)


def get_redis() -> redis.Redis:
    """Return a Redis client backed by the shared connection pool.

    The pool manages connection lifecycle — callers do NOT need to close the client.
    """
    return redis.Redis(connection_pool=_pool)


def check_redis() -> bool:
    """Check Redis connectivity. Returns True if reachable."""
    try:
        return get_redis().ping()
    except redis.ConnectionError:
        return False
    except Exception:
        return False
