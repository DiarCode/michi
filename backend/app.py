"""FastAPI application entry point."""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import init_db
from backend.routers import stations, routes as routes_router, dashboard, alerts, scenarios, analytics
from backend.routers import interventions, executive, depot, passenger_info
from backend.websocket import websocket_router, mock_bus_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    task = asyncio.create_task(mock_bus_stream())
    print("Backend started — DB initialized, bus stream running.")
    yield
    task.cancel()
    print("Shutting down backend...")


app = FastAPI(title="Michi Transit Intelligence API", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

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
app.include_router(websocket_router, prefix="/ws")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
