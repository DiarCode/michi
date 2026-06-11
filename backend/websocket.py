"""WebSocket manager for real-time bus positions, alerts, and simulation events.

Authentication: If WS_AUTH_SECRET env var is set, clients must provide a
matching token as a query parameter (?token=<secret>). Connections without
a valid token are rejected with close code 4001. If WS_AUTH_SECRET is not
set, all connections are allowed (dev mode).
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import WS_AUTH_SECRET
from backend.redis_client import REDIS_URL
from backend.services.realtime_service import get_current_positions

logger = logging.getLogger(__name__)

websocket_router = APIRouter()


def _validate_token(token: str | None) -> bool:
    """Validate a WebSocket connection token.

    If WS_AUTH_SECRET is not set (empty string), allow all connections (dev mode).
    If set, require an exact match.
    """
    if not WS_AUTH_SECRET:
        # Dev mode: no auth required
        return True
    if token is None:
        return False
    # Constant-time comparison to prevent timing attacks
    import hmac

    return hmac.compare_digest(token, WS_AUTH_SECRET)


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, set[str] | None] = {}

    async def connect(self, websocket: WebSocket, subscriptions: set[str] | None = None):
        await websocket.accept()
        self.active_connections[websocket] = subscriptions

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    async def broadcast(self, event_type: str, data: dict):
        disconnected = []
        for ws, subs in self.active_connections.items():
            # Send if: no subscription filter (None) OR event_type in subscriptions
            if subs is None or event_type in subs:
                try:
                    await ws.send_json({"type": event_type, "data": data})
                except Exception:
                    logger.warning("Failed to send WS event '%s' to client, removing connection", event_type)
                    disconnected.append(ws)
        for ws in disconnected:
            self.active_connections.pop(ws, None)


manager = ConnectionManager()


async def bus_stream():
    """Broadcast real-time bus positions every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        positions = get_current_positions()
        for bus in positions:
            await manager.broadcast("bus_position", bus)


async def simulation_relay():
    """Subscribe to Redis michi:simulation channel and broadcast events to WS clients."""
    import redis.asyncio as aioredis

    redis_url = REDIS_URL
    r = aioredis.from_url(redis_url, decode_responses=True)
    pubsub = r.pubsub()
    try:
        await pubsub.subscribe("michi:simulation")
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    event_type = data.get("type", "simulation_tick")
                    await manager.broadcast(event_type, data)
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Invalid simulation relay message received")
    except asyncio.CancelledError:
        return
    finally:
        await pubsub.unsubscribe("michi:simulation")
        await r.close()


async def ml_relay():
    """Subscribe to Redis michi:ml channel and broadcast ML events to WS clients.

    Handles drift_alert, retrain_started, retrain_progress, retrain_completed,
    model_promoted, and other ML lifecycle events.
    """
    import redis.asyncio as aioredis

    redis_url = REDIS_URL
    r = aioredis.from_url(redis_url, decode_responses=True)
    pubsub = r.pubsub()
    try:
        await pubsub.subscribe("michi:ml")
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    event_type = data.get("type", "ml_event")
                    await manager.broadcast(event_type, data)
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Invalid ML relay message received")
    except asyncio.CancelledError:
        return
    finally:
        await pubsub.unsubscribe("michi:ml")
        await r.close()


async def combined_stream():
    """Run bus position stream, simulation relay, and ML relay concurrently."""
    bus_task = asyncio.create_task(bus_stream())
    sim_task = asyncio.create_task(simulation_relay())
    ml_task = asyncio.create_task(ml_relay())
    try:
        await asyncio.gather(bus_task, sim_task, ml_task)
    except asyncio.CancelledError:
        bus_task.cancel()
        sim_task.cancel()
        ml_task.cancel()
        await asyncio.gather(bus_task, sim_task, ml_task, return_exceptions=True)


@websocket_router.websocket("/realtime")
async def realtime_ws(websocket: WebSocket):
    """Real-time WebSocket endpoint for bus positions, alerts, and simulation events.

    Authentication: optional ?token=<secret> query param. If WS_AUTH_SECRET env
    var is set, the token must match. Otherwise, dev mode (no auth required).
    """
    # Extract token from query parameters
    token = websocket.query_params.get("token")
    if not _validate_token(token):
        await websocket.close(code=4001, reason="Unauthorized: invalid or missing token")
        logger.warning(
            "WebSocket connection rejected: invalid token from %s",
            websocket.client.host if websocket.client else "unknown",
        )
        return

    await manager.connect(websocket)
    logger.info("WebSocket client connected (auth=%s)", "token" if token else "dev-mode")
    try:
        while True:
            data = await websocket.receive_json()
            if "subscribe" in data:
                manager.active_connections[websocket] = set(data["subscribe"])
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
