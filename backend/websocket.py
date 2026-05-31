"""WebSocket manager for real-time bus positions, alerts, and simulation events."""

import asyncio
import json
import os
from typing import Dict, Set, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.realtime_service import BUS_POOL, get_current_positions

websocket_router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, Optional[Set[str]]] = {}

    async def connect(self, websocket: WebSocket, subscriptions: Optional[Set[str]] = None):
        await websocket.accept()
        self.active_connections[websocket] = subscriptions  # None means all events

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    async def broadcast(self, event_type: str, data: Dict):
        disconnected = []
        for ws, subs in self.active_connections.items():
            # Send if: no subscription filter (None) OR event_type in subscriptions
            if subs is None or event_type in subs:
                try:
                    await ws.send_json({"type": event_type, **data})
                except Exception:
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
            await manager.broadcast("bus_position", {"data": bus})


async def simulation_relay():
    """Subscribe to Redis michi:simulation channel and broadcast events to WS clients."""
    import redis.asyncio as aioredis

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
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
                    pass
    except asyncio.CancelledError:
        return
    finally:
        await pubsub.unsubscribe("michi:simulation")
        await r.close()


async def combined_stream():
    """Run bus position stream and simulation relay concurrently."""
    bus_task = asyncio.create_task(bus_stream())
    sim_task = asyncio.create_task(simulation_relay())
    try:
        await asyncio.gather(bus_task, sim_task)
    except asyncio.CancelledError:
        bus_task.cancel()
        sim_task.cancel()
        await asyncio.gather(bus_task, sim_task, return_exceptions=True)


@websocket_router.websocket("realtime")
async def realtime_ws(websocket: WebSocket):
    await manager.connect(websocket)  # default: all events
    try:
        while True:
            data = await websocket.receive_json()
            if "subscribe" in data:
                manager.active_connections[websocket] = set(data["subscribe"])
    except WebSocketDisconnect:
        manager.disconnect(websocket)