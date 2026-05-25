"""WebSocket manager for real-time bus positions and alerts."""

import asyncio
import json
import random
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

websocket_router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        text = json.dumps(message)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(text)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

MOCK_BUSES = [
    {"bus_id": "BUS-001", "route_id": "Route_12", "lat": 51.1605, "lon": 71.4702},
    {"bus_id": "BUS-002", "route_id": "Route_34", "lat": 51.1450, "lon": 71.4300},
]

async def mock_bus_stream():
    """Broadcast mock bus positions every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        for bus in MOCK_BUSES:
            bus["lat"] += random.uniform(-0.001, 0.001)
            bus["lon"] += random.uniform(-0.001, 0.001)
            bus["speed_kmh"] = random.randint(15, 55)
            bus["occupancy_percent"] = random.randint(20, 95)
            bus["next_stop"] = random.choice(["Khan Shatyr", "Mega Silk Way", "Nurzhol Blvd"])
            bus["eta_seconds"] = random.randint(30, 300)
            await manager.broadcast({
                "type": "bus_position",
                "data": bus,
            })

@websocket_router.websocket("realtime")
async def realtime_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"type": "ack", "data": data}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
