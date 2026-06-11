"""Shared configuration — single source of truth for environment variables."""

import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./michi.db")
WS_AUTH_SECRET = os.getenv("WS_AUTH_SECRET", "")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3100,http://localhost:5173,http://localhost:8600,http://localhost:8000,http://localhost:8100",
)
