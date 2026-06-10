"""FastAPI application entry point."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.database import init_db
from backend.exceptions import AppException
from backend.routers import (
    alerts,
    analytics,
    dashboard,
    depot,
    executive,
    interventions,
    passenger_info,
    scenarios,
    simulation,
    stations,
    timeline,
)
from backend.routers import routes as routes_router
from backend.websocket import combined_stream, websocket_router

logger = logging.getLogger("michi")

# Configure allowed CORS origins from environment variable.
# Comma-separated list of origins. Defaults to localhost dev servers.
# Set ALLOWED_ORIGINS=* to revert to permissive mode (not recommended for production).
_DEFAULT_ORIGINS = "http://localhost:3100,http://localhost:5173,http://localhost:8600,http://localhost:8000,http://localhost:8100"


def _parse_allowed_origins() -> list[str]:
    """Parse ALLOWED_ORIGINS env var into a list of allowed origins."""
    raw = os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Warm-load the DTS-GSSF model into cache
    try:
        from backend.ml.predictor import get_cached_model

        model, normalizer = get_cached_model()
        if model:
            norm_status = "with z-score normalization" if normalizer else "WITHOUT normalization"
            logger.info("DTS-GSSF model loaded and cached (%s).", norm_status)
        else:
            logger.info("No production model artifact found — using mock predictions.")
    except Exception as e:
        logger.warning("Model warm-load failed: %s — using mock predictions.", e)
    # Auto-start simulation (non-blocking; the Celery task runs independently)
    try:
        from backend.tasks import celery_app

        celery_app.send_task("run_simulation")
        logger.info("Simulation task dispatched.")
    except Exception as e:
        logger.info("Simulation auto-start skipped: %s", e)
    stream_task = asyncio.create_task(combined_stream())
    logger.info("Backend started — DB initialized, bus stream + simulation relay running.")
    yield
    stream_task.cancel()
    logger.info("Shutting down backend...")


app = FastAPI(title="Michi Transit Intelligence API", version="1.0.0", lifespan=lifespan)

# CORS configuration from environment variable
allowed_origins = _parse_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler for unhandled errors
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": 500},
    )


# Handler for application-level exceptions with proper status codes
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status": exc.status_code, "error_code": exc.error_code},
    )


app.include_router(stations.router, prefix="/api/v1/stations", tags=["stations"])
app.include_router(routes_router.router, prefix="/api/v1/routes", tags=["routes"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(scenarios.router, prefix="/api/v1/scenarios", tags=["scenarios"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(interventions.router, prefix="/api/v1/interventions", tags=["interventions"])
app.include_router(executive.router, prefix="/api/v1/executive", tags=["executive"])
app.include_router(depot.router, prefix="/api/v1/depot", tags=["depot"])
app.include_router(passenger_info.router, prefix="/api/v1/passenger", tags=["passenger"])
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["simulation"])
app.include_router(timeline.router, prefix="/api/v1/timeline", tags=["timeline"])
app.include_router(websocket_router, prefix="/ws")


@app.get("/health")
def health_check():
    """Enriched health check with DB, model, and Redis status."""
    checks: dict[str, str] = {}

    # Database connectivity
    try:
        from backend.database import SessionLocal
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # Model cache status
    try:
        from backend.ml.predictor import get_cached_model
        model, normalizer = get_cached_model()
        checks["model"] = "loaded" if model else "not_loaded"
        checks["normalizer"] = "loaded" if normalizer else "not_loaded"
    except Exception as e:
        checks["model"] = f"error: {e}"

    # Redis connectivity (optional — may not be configured)
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        r.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"

    overall = "ok" if all(v == "ok" or v == "loaded" or v == "not_loaded" for v in checks.values()) else "degraded"
    return {"status": overall, "version": "1.0.0", "checks": checks}
