# Michi Platform Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Transform the DTS-GSSF research prototype into a production-grade Astana Transit Intelligence Platform with real OSM data, FastAPI backend, React dashboard, and Docker deployment.

**Architecture:** Hybrid — React operational dashboard (real-time tracking, scenario planning) + enhanced Streamlit research UI (model training, evaluation). FastAPI backend with PostgreSQL/TimescaleDB, Redis, Celery. Real Astana bus network from OpenStreetMap.

**Tech Stack:** Python 3.13, PyTorch, FastAPI, SQLAlchemy, PostgreSQL, TimescaleDB, Redis, Celery, React 19, Vite, TypeScript, shadcn/ui, Tailwind CSS, TanStack Query, Zustand, Recharts, MapCN, Docker, Docker Compose.

---

## File Structure Summary

**New directories:**
- `backend/` — FastAPI application
- `dashboard/` — React operational dashboard
- `data/` — OSM parser, download scripts, cache
- `tests/` — Unit tests

**Modified files:**
- `main.py` — Add real data integration
- `pyproject.toml` — Add backend dependencies
- `README.md` — Update documentation

**New files:**
- `docker-compose.yml` — Full stack orchestration
- `backend/Dockerfile`, `dashboard/Dockerfile`, `Dockerfile` — Service containers

---

## Sub-Project 1: Real Data Foundation

### Task 1.1: Add OSM dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1:** Add OSM parsing dependencies.

```toml
dependencies = [
    "numpy>=2.4.1",
    "pandas>=2.3.3",
    "plotly>=6.5.1",
    "statsmodels>=0.14.6",
    "streamlit>=1.52.2",
    "torch>=2.9.1",
    "watchdog>=6.0.0",
    "xgboost>=3.1.3",
    "osmnx>=2.0.0",
    "requests>=2.32.0",
    "shapely>=2.0.0",
]
```

**Step 2:** Run `uv sync` to install new dependencies.

Run: `uv sync`
Expected: Dependencies installed successfully.

**Step 3:** Commit.

```bash
git add pyproject.toml
git commit -m "chore: add OSM parsing dependencies"
```

---

### Task 1.2: Create OSM download script

**Files:**
- Create: `data/download_osm.py`

**Step 1:** Write the download script.

```python
"""Download and cache Kazakhstan OSM extract for Astana parsing."""

import os
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OSM_URL = "https://download.geofabrik.de/asia/kazakhstan-latest.osm.pbf"
CACHE_FILE = CACHE_DIR / "kazakhstan-latest.osm.pbf"

ASTANA_BBOX = {
    "west": 71.25,
    "east": 71.65,
    "south": 50.95,
    "north": 51.25,
}


def download_osm(force: bool = False) -> Path:
    """Download Kazakhstan OSM PBF if not cached or stale."""
    if CACHE_FILE.exists() and not force:
        age_days = (Path.now() - CACHE_FILE.stat().st_mtime).days
        if age_days < 30:
            print(f"Using cached OSM file: {CACHE_FILE}")
            return CACHE_FILE

    print(f"Downloading {OSM_URL}...")
    try:
        urlretrieve(OSM_URL, CACHE_FILE)
        print(f"Downloaded to {CACHE_FILE}")
        return CACHE_FILE
    except URLError as e:
        print(f"Failed to download OSM: {e}")
        if CACHE_FILE.exists():
            print("Falling back to cached file")
            return CACHE_FILE
        raise


def extract_astana_bbox(input_path: Path, output_path: Path) -> Path:
    """Extract Astana bounding box using osmium (if available)."""
    import shutil
    import subprocess

    if not shutil.which("osmium"):
        print("osmium not found — skipping bbox extraction, will filter programmatically")
        return input_path

    bbox_str = f"{ASTANA_BBOX['west']},{ASTANA_BBOX['south']},{ASTANA_BBOX['east']},{ASTANA_BBOX['north']}"
    cmd = [
        "osmium", "extract",
        "--bbox", bbox_str,
        "--set-bounds",
        "--overwrite",
        "-o", str(output_path),
        str(input_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


if __name__ == "__main__":
    download_osm()
```

**Step 2:** Run the download script.

Run: `uv run python data/download_osm.py`
Expected: Either downloads the file or reports caching.

**Step 3:** Commit.

```bash
git add data/download_osm.py
git commit -m "feat: add OSM download script with caching"
```

---

### Task 1.3: Create OSM parser

**Files:**
- Create: `data/osm_parser.py`

**Step 1:** Write the OSM parser.

```python
"""Parse Astana bus network from OSM data."""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import osmnx as ox
import pandas as pd
from shapely.geometry import Point

CACHE_DIR = Path(__file__).parent / "cache"

# District polygons (simplified bounding boxes for Astana districts)
DISTRICTS = {
    "Esil": {"west": 71.35, "east": 71.55, "south": 51.10, "north": 51.25},
    "Almaty": {"west": 71.40, "east": 71.55, "south": 51.00, "north": 51.12},
    "Saryarka": {"west": 71.30, "east": 71.50, "south": 50.95, "north": 51.08},
    "Baikonur": {"west": 71.25, "east": 71.45, "south": 51.05, "north": 51.20},
}


def _point_in_bbox(lat: float, lon: float, bbox: Dict[str, float]) -> bool:
    return bbox["west"] <= lon <= bbox["east"] and bbox["south"] <= lat <= bbox["north"]


def assign_district(lat: float, lon: float) -> Optional[str]:
    for name, bbox in DISTRICTS.items():
        if _point_in_bbox(lat, lon, bbox):
            return name
    return None


def parse_astana_bus_network(osm_file: Path) -> Dict:
    """Parse bus routes and stops from OSM PBF file."""
    print(f"Parsing OSM file: {osm_file}")

    # Use osmnx to query the OSM file for bus routes
    # osmnx can read .osm.pbf files directly
    try:
        G = ox.graph_from_xml(str(osm_file), retain_all=True)
    except Exception as e:
        print(f"osmnx parse failed: {e}")
        return {}

    # Query bus routes using overpass-style filtering
    # For osmnx graph data, we need to work with nodes/edges
    nodes, edges = ox.graph_to_gdfs(G)

    # Filter for bus stops (nodes with highway=bus_stop)
    bus_stop_mask = nodes.get("highway", "") == "bus_stop"
    bus_stops = nodes[bus_stop_mask].copy()

    if bus_stops.empty:
        print("No bus stops found in OSM data")
        return {}

    # Build stops list
    stops = []
    for idx, row in bus_stops.iterrows():
        lat, lon = row["y"], row["x"]
        name = row.get("name", f"Stop {idx}")
        district = assign_district(lat, lon)
        stops.append({
            "stop_id": str(idx),
            "name": name if pd.notna(name) else f"Stop {idx}",
            "lat": float(lat),
            "lon": float(lon),
            "district": district or "Unknown",
        })

    # Build routes from edges with route=bus
    route_mask = edges.get("route", "") == "bus"
    bus_edges = edges[route_mask]

    routes = []
    if not bus_edges.empty:
        # Group by route name/ref
        route_groups = bus_edges.groupby("ref" if "ref" in bus_edges.columns else "name")
        for route_name, group in route_groups:
            if pd.isna(route_name):
                continue
            route_nodes = set()
            for _, edge in group.iterrows():
                route_nodes.add(edge["u"])
                route_nodes.add(edge["v"])

            # Order stops by geometry
            route_stops = [s for s in stops if s["stop_id"] in route_nodes]
            route_stops.sort(key=lambda s: s["lat"])

            routes.append({
                "route_id": str(route_name).replace(" ", "_"),
                "name": f"Route {route_name}",
                "color": "#2E86AB",
                "stop_ids": [s["stop_id"] for s in route_stops],
            })

    # Build adjacency: consecutive stops on same route are connected
    adjacency = {}
    for route in routes:
        for i in range(len(route["stop_ids"]) - 1):
            a, b = route["stop_ids"][i], route["stop_ids"][i + 1]
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

    result = {
        "stops": stops,
        "routes": routes,
        "adjacency": adjacency,
        "metadata": {
            "source": "osm",
            "bbox": {"west": 71.25, "east": 71.65, "south": 50.95, "north": 51.25},
        },
    }

    print(f"Parsed {len(stops)} stops, {len(routes)} routes")
    return result


def save_parsed_network(network: Dict, output_dir: Path = CACHE_DIR):
    output_dir.mkdir(exist_ok=True)
    for key in ["stops", "routes", "adjacency"]:
        path = output_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(network.get(key, {}), f, indent=2, ensure_ascii=False)
    print(f"Saved parsed network to {output_dir}")


def load_parsed_network(input_dir: Path = CACHE_DIR) -> Optional[Dict]:
    try:
        network = {}
        for key in ["stops", "routes", "adjacency"]:
            path = input_dir / f"{key}.json"
            with open(path, "r", encoding="utf-8") as f:
                network[key] = json.load(f)
        return network
    except FileNotFoundError:
        return None


def get_astana_network(use_cache: bool = True) -> Dict:
    """Get parsed Astana bus network, downloading and parsing if needed."""
    if use_cache:
        cached = load_parsed_network()
        if cached:
            return cached

    from .download_osm import download_osm
    osm_file = download_osm()
    network = parse_astana_bus_network(osm_file)
    if network:
        save_parsed_network(network)
    return network


if __name__ == "__main__":
    network = get_astana_network(use_cache=False)
    print(f"Network: {len(network.get('stops', []))} stops, {len(network.get('routes', []))} routes")
```

**Step 2:** Run the parser.

Run: `uv run python data/osm_parser.py`
Expected: Parses OSM data and outputs stop/route counts.

**Step 3:** Commit.

```bash
git add data/osm_parser.py
git commit -m "feat: add OSM parser for Astana bus network"
```

---

### Task 1.4: Integrate real network into main.py

**Files:**
- Modify: `main.py`

**Step 1:** Add `use_real_data` parameter to `build_astana_network()`.

Find `def build_astana_network()` and modify to:

```python
def build_astana_network(use_real_data: bool = False) -> NetworkSpec:
    """Build Astana bus network — real OSM data or synthetic fallback."""
    if use_real_data:
        try:
            from data.osm_parser import get_astana_network
            network = get_astana_network()
            if network and len(network.get("stops", [])) >= 5:
                stops = network["stops"]
                routes = network["routes"]
                adjacency = network["adjacency"]
                n_stations = len(stops)
                n_lines = len(routes)

                station_names = [s["name"] for s in stops]
                station_district = [s.get("district", "Unknown") for s in stops]
                lines = [r["stop_ids"] for r in routes]

                # Build adjacency matrix
                A_phys = np.zeros((n_stations, n_stations))
                for i, stop in enumerate(stops):
                    neighbors = adjacency.get(stop["stop_id"], [])
                    for neighbor_id in neighbors:
                        for j, other in enumerate(stops):
                            if other["stop_id"] == neighbor_id:
                                A_phys[i, j] = 1.0
                                break

                edges = []
                for i in range(n_stations):
                    for j in range(i + 1, n_stations):
                        if A_phys[i, j] > 0:
                            edges.append((i, j))

                print(f"Using real OSM network: {n_stations} stations, {n_lines} lines")
                return NetworkSpec(
                    station_names=station_names,
                    station_district=station_district,
                    lines=lines,
                    A_phys=A_phys,
                    edges=edges,
                )
        except Exception as e:
            print(f"Real data loading failed: {e}. Falling back to synthetic.")

    # --- existing synthetic code ---
    # (keep the original synthetic implementation as fallback)
    # ... original code ...
```

**Step 2:** Add real data toggle to Streamlit UI.

Find the Streamlit setup section and add:

```python
use_real_data = st.sidebar.checkbox("Use real OSM data", value=False)
network = build_astana_network(use_real_data=use_real_data)
```

**Step 3:** Run a quick test.

Run: `uv run python -c "from main import build_astana_network; n = build_astana_network(use_real_data=False); print(n)"`
Expected: Synthetic network loads successfully.

**Step 4:** Commit.

```bash
git add main.py
git commit -m "feat: integrate real OSM data into network builder"
```

---

### Task 1.5: Write tests for OSM parser

**Files:**
- Create: `tests/test_osm_parser.py`

**Step 1:** Write tests.

```python
import json
import tempfile
from pathlib import Path

import pytest

from data.osm_parser import parse_astana_bus_network, assign_district, save_parsed_network, load_parsed_network


class TestAssignDistrict:
    def test_esil_district(self):
        assert assign_district(51.15, 71.40) == "Esil"

    def test_saryarka_district(self):
        assert assign_district(51.00, 71.35) == "Saryarka"

    def test_outside_astana(self):
        assert assign_district(0.0, 0.0) is None


class TestParseAstanaBusNetwork:
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".osm", delete=False) as f:
            f.write(b"<osm></osm>")
            f.flush()
            result = parse_astana_bus_network(Path(f.name))
        assert result == {}


class TestSaveLoad:
    def test_roundtrip(self):
        network = {
            "stops": [{"stop_id": "1", "name": "Test", "lat": 51.0, "lon": 71.0}],
            "routes": [],
            "adjacency": {},
        }
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            save_parsed_network(network, d)
            loaded = load_parsed_network(d)
        assert loaded["stops"][0]["name"] == "Test"
```

**Step 2:** Run tests.

Run: `pytest tests/test_osm_parser.py -v`
Expected: All tests pass.

**Step 3:** Commit.

```bash
git add tests/test_osm_parser.py
git commit -m "test: add OSM parser unit tests"
```

---

## Sub-Project 2: Professional Backend & API

### Task 2.1: Create FastAPI app structure

**Files:**
- Create: `backend/app.py`
- Create: `backend/__init__.py`

**Step 1:** Write the FastAPI entry point.

```python
"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import stations, routes as routes_router, dashboard, alerts, scenarios
from backend.websocket import websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting backend...")
    yield
    # Shutdown
    print("Shutting down backend...")


app = FastAPI(
    title="Michi Transit Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stations.router, prefix="/api/v1/stations", tags=["stations"])
app.include_router(routes_router.router, prefix="/api/v1/routes", tags=["routes"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(scenarios.router, prefix="/api/v1/scenarios", tags=["scenarios"])
app.include_router(websocket_router, prefix="/ws")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
```

**Step 2:** Create router stubs.

Create `backend/routers/__init__.py` and individual router files with basic CRUD.

```python
# backend/routers/stations.py
from fastapi import APIRouter

router = APIRouter()

@router.get("")
def list_stations():
    return {"stations": []}

@router.get("/{station_id}/forecast")
def get_station_forecast(station_id: str):
    return {"station_id": station_id, "forecast": []}
```

Create similar stubs for `routes.py`, `dashboard.py`, `alerts.py`, `scenarios.py`.

**Step 3:** Commit.

```bash
git add backend/
git commit -m "feat: scaffold FastAPI backend with router structure"
```

---

### Task 2.2: Create database models and setup

**Files:**
- Create: `backend/database.py`
- Create: `backend/models.py`

**Step 1:** Write database setup.

```python
# backend/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/michi")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Step 2:** Write Pydantic models.

```python
# backend/models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Station(BaseModel):
    id: int
    stop_id: str
    name: str
    lat: float
    lon: float
    district: Optional[str] = None
    route_ids: List[int] = []


class Route(BaseModel):
    id: int
    route_id: str
    name: str
    color: Optional[str] = None
    stop_sequence: List[int] = []


class Alert(BaseModel):
    id: int
    alert_type: str
    severity: str
    station_id: Optional[int] = None
    route_id: Optional[int] = None
    message: str
    acknowledged: bool = False
    created_at: datetime


class KPIDashboard(BaseModel):
    total_stations: int
    active_routes: int
    avg_ridership: float
    alerts_today: int
```

**Step 3:** Commit.

```bash
git add backend/database.py backend/models.py
git commit -m "feat: add database setup and Pydantic models"
```

---

### Task 2.3: Create WebSocket manager with mock bus positions

**Files:**
- Create: `backend/websocket.py`

**Step 1:** Write the WebSocket manager.

```python
"""WebSocket manager for real-time bus positions and alerts."""

import asyncio
import json
import random
from typing import Dict, List
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

websocket_router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        text = json.dumps(message)
        for connection in self.active_connections:
            await connection.send_text(text)


manager = ConnectionManager()

# Mock bus positions
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
            # Echo back for now
            await websocket.send_text(json.dumps({"type": "ack", "data": data}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

**Step 2:** Commit.

```bash
git add backend/websocket.py
git commit -m "feat: add WebSocket manager with mock bus stream"
```

---

### Task 2.4: Create Celery tasks placeholder

**Files:**
- Create: `backend/tasks.py`

**Step 1:** Write Celery configuration.

```python
"""Celery background tasks."""

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("michi", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Almaty",
    enable_utc=True,
    beat_schedule={
        "generate-forecasts": {
            "task": "backend.tasks.generate_forecasts",
            "schedule": 900.0,  # every 15 minutes
        },
    },
)


@celery_app.task
def generate_forecasts():
    """Generate forecasts for all stations."""
    print("Generating forecasts...")
    return {"status": "ok"}


@celery_app.task
def retrain_model():
    """Retrain the DTS-GSSF model."""
    print("Retraining model...")
    return {"status": "ok"}
```

**Step 2:** Commit.

```bash
git add backend/tasks.py
git commit -m "feat: add Celery background tasks"
```

---

### Task 2.5: Add backend dependencies and Dockerfile

**Files:**
- Modify: `pyproject.toml`
- Create: `backend/Dockerfile`
- Create: `backend/requirements.txt`

**Step 1:** Add backend dependencies to pyproject.toml.

```toml
dependencies = [
    # ... existing deps ...
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "redis>=5.0.0",
    "celery>=5.4.0",
    "pydantic>=2.10.0",
]
```

**Step 2:** Create backend Dockerfile.

```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 3:** Create backend requirements.txt.

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
redis>=5.0.0
celery>=5.4.0
pydantic>=2.10.0
numpy>=2.4.1
pandas>=2.3.3
torch>=2.9.1
requests>=2.32.0
```

**Step 4:** Commit.

```bash
git add pyproject.toml backend/Dockerfile backend/requirements.txt
git commit -m "chore: add backend dependencies and Dockerfile"
```

---

## Sub-Project 3: Operational Dashboard (React + MapCN)

### Task 3.1: Initialize React project with Vite

**Files:**
- Create: `dashboard/` directory and all scaffold files

**Step 1:** Initialize Vite project.

Run:
```bash
cd dashboard && npm create vite@latest . -- --template react-ts
```
Expected: Vite scaffold created.

**Step 2:** Install dependencies.

Run:
```bash
cd dashboard && npm install
npm install react-router-dom zustand @tanstack/react-query axios recharts lucide-react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Step 3:** Configure Tailwind.

`tailwind.config.js`:
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

`src/styles/globals.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Step 4:** Commit.

```bash
git add dashboard/
git commit -m "feat: initialize React dashboard with Vite + Tailwind"
```

---

### Task 3.2: Set up shadcn/ui

**Files:**
- Modify: `dashboard/`

**Step 1:** Initialize shadcn/ui.

Run:
```bash
cd dashboard && npx shadcn-ui@latest init -y
```

**Step 2:** Install common components.

Run:
```bash
cd dashboard && npx shadcn-ui@latest add button card table badge alert dialog tabs
```

**Step 3:** Commit.

```bash
git add dashboard/
git commit -m "feat: add shadcn/ui components"
```

---

### Task 3.3: Create API client and types

**Files:**
- Create: `dashboard/src/lib/api.ts`
- Create: `dashboard/src/lib/utils.ts`
- Create: `dashboard/src/types/index.ts`

**Step 1:** Write API client.

```typescript
// dashboard/src/lib/api.ts
import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1",
});

export const fetchStations = () => api.get("/stations").then((r) => r.data);
export const fetchRoutes = () => api.get("/routes").then((r) => r.data);
export const fetchKPIs = () => api.get("/dashboard/kpis").then((r) => r.data);
export const fetchAlerts = () => api.get("/alerts").then((r) => r.data);
```

**Step 2:** Write types.

```typescript
// dashboard/src/types/index.ts
export interface Station {
  id: number;
  stop_id: string;
  name: string;
  lat: number;
  lon: number;
  district?: string;
}

export interface Route {
  id: number;
  route_id: string;
  name: string;
  color?: string;
  stop_sequence: number[];
}

export interface Alert {
  id: number;
  alert_type: string;
  severity: "low" | "medium" | "high" | "critical";
  message: string;
  created_at: string;
}

export interface BusPosition {
  bus_id: string;
  route_id: string;
  lat: number;
  lon: number;
  speed_kmh: number;
  next_stop: string;
  eta_seconds: number;
  occupancy_percent: number;
}
```

**Step 3:** Commit.

```bash
git add dashboard/src/lib/api.ts dashboard/src/types/index.ts
git commit -m "feat: add API client and TypeScript types"
```

---

### Task 3.4: Create layout components

**Files:**
- Create: `dashboard/src/components/layout/Sidebar.tsx`
- Create: `dashboard/src/components/layout/TopBar.tsx`
- Create: `dashboard/src/components/layout/RoleGuard.tsx`

**Step 1:** Write Sidebar.

```tsx
// dashboard/src/components/layout/Sidebar.tsx
import { NavLink } from "react-router-dom";
import { Map, BarChart, AlertTriangle, Settings, FlaskConical } from "lucide-react";

const navItems = [
  { to: "/", label: "Command Center", icon: BarChart },
  { to: "/map", label: "Live Map", icon: Map },
  { to: "/alerts", label: "Alerts", icon: AlertTriangle },
  { to: "/scenarios", label: "Scenarios", icon: FlaskConical },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  return (
    <aside className="w-64 h-screen bg-slate-900 text-white flex flex-col">
      <div className="p-4 text-xl font-bold">Michi</div>
      <nav className="flex-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 hover:bg-slate-800 transition ${
                isActive ? "bg-slate-800 border-l-4 border-blue-500" : ""
              }`
            }
          >
            <item.icon size={20} />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
```

**Step 2:** Write TopBar.

```tsx
// dashboard/src/components/layout/TopBar.tsx
import { useState } from "react";

const ROLES = ["Dispatch Manager", "City Planner", "Executive"];

export default function TopBar() {
  const [role, setRole] = useState("Dispatch Manager");

  return (
    <header className="h-16 bg-white border-b flex items-center justify-between px-6">
      <h1 className="text-lg font-semibold">Astana Transit Intelligence</h1>
      <div className="flex items-center gap-4">
        <span className="text-sm text-gray-500">Role:</span>
        <select
          value={role}
          onChange={(e) => setRole(e.target.value)}
          className="border rounded px-2 py-1 text-sm"
        >
          {ROLES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>
    </header>
  );
}
```

**Step 3:** Commit.

```bash
git add dashboard/src/components/layout/
git commit -m "feat: add dashboard layout components"
```

---

### Task 3.5: Create MapCN container and map components

**Files:**
- Create: `dashboard/src/components/map/MapContainer.tsx`
- Create: `dashboard/src/components/map/StationMarker.tsx`
- Create: `dashboard/src/components/map/BusMarker.tsx`

**Step 1:** Write MapContainer with iframe embedding.

```tsx
// dashboard/src/components/map/MapContainer.tsx
import { useEffect, useRef } from "react";
import type { Station, BusPosition } from "@/types";

interface Props {
  stations: Station[];
  buses: BusPosition[];
}

export default function MapContainer({ stations, buses }: Props) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const center = stations.length > 0
    ? `${stations[0].lat},${stations[0].lon}`
    : "51.1605,71.4702";

  const markersParam = stations
    .map((s) => `${s.lat},${s.lon},${encodeURIComponent(s.name)}`)
    .join("|");

  return (
    <div className="relative w-full h-full">
      <iframe
        ref={iframeRef}
        src={`https://mapcn.dev/embed?center=${center}&zoom=13&markers=${markersParam}`}
        className="w-full h-full border-0"
        allow="geolocation"
        title="Astana Map"
      />
      {/* Overlay for bus markers */}
      <div className="absolute top-4 left-4 bg-white/90 p-3 rounded shadow">
        <h3 className="font-bold text-sm">Active Buses</h3>
        <p className="text-xs text-gray-600">{buses.length} buses tracked</p>
      </div>
    </div>
  );
}
```

**Step 2:** Commit.

```bash
git add dashboard/src/components/map/
git commit -m "feat: add MapCN container with iframe embedding"
```

---

### Task 3.6: Create dashboard pages

**Files:**
- Create: `dashboard/src/routes/CommandCenter.tsx`
- Create: `dashboard/src/routes/LiveMap.tsx`
- Create: `dashboard/src/routes/AlertsPage.tsx`
- Create: `dashboard/src/routes/ScenarioPlanner.tsx`
- Create: `dashboard/src/routes/Settings.tsx`

**Step 1:** Write CommandCenter page.

```tsx
// dashboard/src/routes/CommandCenter.tsx
import { useQuery } from "@tanstack/react-query";
import { fetchKPIs, fetchStations } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function CommandCenter() {
  const { data: kpis } = useQuery({ queryKey: ["kpis"], queryFn: fetchKPIs });
  const { data: stations } = useQuery({ queryKey: ["stations"], queryFn: fetchStations });

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Command Center</h2>
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-gray-500">Stations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{kpis?.total_stations ?? "—"}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-gray-500">Active Routes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{kpis?.active_routes ?? "—"}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-gray-500">Avg Ridership</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{kpis?.avg_ridership ?? "—"}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-gray-500">Alerts Today</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{kpis?.alerts_today ?? "—"}</div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```

**Step 2:** Write LiveMap page.

```tsx
// dashboard/src/routes/LiveMap.tsx
import { useEffect, useState } from "react";
import MapContainer from "@/components/map/MapContainer";
import type { Station, BusPosition } from "@/types";

export default function LiveMap() {
  const [stations, setStations] = useState<Station[]>([]);
  const [buses, setBuses] = useState<BusPosition[]>([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/realtime");
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "bus_position") {
        setBuses((prev) => {
          const filtered = prev.filter((b) => b.bus_id !== msg.data.bus_id);
          return [...filtered, msg.data];
        });
      }
    };
    return () => ws.close();
  }, []);

  return (
    <div className="h-[calc(100vh-4rem)]">
      <MapContainer stations={stations} buses={buses} />
    </div>
  );
}
```

**Step 3:** Write AlertsPage.

```tsx
// dashboard/src/routes/AlertsPage.tsx
import { useQuery } from "@tanstack/react-query";
import { fetchAlerts } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { AlertTriangle } from "lucide-react";

export default function AlertsPage() {
  const { data } = useQuery({ queryKey: ["alerts"], queryFn: fetchAlerts });
  const alerts = data?.alerts ?? [];

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Alerts</h2>
      <div className="space-y-3">
        {alerts.map((alert: any) => (
          <Card key={alert.id}>
            <CardContent className="flex items-center gap-4 p-4">
              <AlertTriangle className="text-red-500" />
              <div>
                <div className="font-semibold">{alert.alert_type}</div>
                <div className="text-sm text-gray-600">{alert.message}</div>
              </div>
            </CardContent>
          </Card>
        ))}
        {alerts.length === 0 && <p className="text-gray-500">No alerts.</p>}
      </div>
    </div>
  );
}
```

**Step 4:** Write App.tsx with routing.

```tsx
// dashboard/src/App.tsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Sidebar from "@/components/layout/Sidebar";
import TopBar from "@/components/layout/TopBar";
import CommandCenter from "@/routes/CommandCenter";
import LiveMap from "@/routes/LiveMap";
import AlertsPage from "@/routes/AlertsPage";
import ScenarioPlanner from "@/routes/ScenarioPlanner";
import Settings from "@/routes/Settings";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex h-screen">
          <Sidebar />
          <div className="flex-1 flex flex-col">
            <TopBar />
            <main className="flex-1 overflow-auto bg-gray-50">
              <Routes>
                <Route path="/" element={<CommandCenter />} />
                <Route path="/map" element={<LiveMap />} />
                <Route path="/alerts" element={<AlertsPage />} />
                <Route path="/scenarios" element={<ScenarioPlanner />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

**Step 5:** Commit.

```bash
git add dashboard/src/routes/ dashboard/src/App.tsx
git commit -m "feat: add dashboard pages and routing"
```

---

### Task 3.7: Add dashboard Dockerfile

**Files:**
- Create: `dashboard/Dockerfile`

**Step 1:** Write Dockerfile.

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
```

**Step 2:** Create nginx.conf.

```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

**Step 3:** Commit.

```bash
git add dashboard/Dockerfile dashboard/nginx.conf
git commit -m "feat: add dashboard Dockerfile and nginx config"
```

---

## Sub-Project 4: Scenario Simulator

### Task 4.1: Create backend scenario engine

**Files:**
- Create: `backend/routers/scenarios.py`
- Create: `backend/services/scenario_engine.py`

**Step 1:** Write scenario router.

```python
# backend/routers/scenarios.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()


class ScenarioConfig(BaseModel):
    name: str
    modifications: List[Dict[str, Any]]


class ScenarioResult(BaseModel):
    scenario_id: str
    base_metrics: Dict[str, float]
    scenario_metrics: Dict[str, float]
    changes: Dict[str, float]


@router.post("/run")
def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    # Placeholder: will integrate with DTS-GSSF model
    return ScenarioResult(
        scenario_id="scen-001",
        base_metrics={"ridership": 10000, "avg_wait": 5.2},
        scenario_metrics={"ridership": 9500, "avg_wait": 4.8},
        changes={"ridership": -5.0, "avg_wait": -7.7},
    )
```

**Step 2:** Commit.

```bash
git add backend/routers/scenarios.py
git commit -m "feat: add scenario simulator backend router"
```

---

### Task 4.2: Create frontend scenario UI

**Files:**
- Modify: `dashboard/src/routes/ScenarioPlanner.tsx`

**Step 1:** Write scenario planner UI.

```tsx
// dashboard/src/routes/ScenarioPlanner.tsx
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

export default function ScenarioPlanner() {
  const [result, setResult] = useState<any>(null);

  const runScenario = async () => {
    const { data } = await api.post("/scenarios/run", {
      name: "Frequency Increase",
      modifications: [{ type: "frequency", target: "Route_12", params: { headway: 5 } }],
    });
    setResult(data);
  };

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Scenario Planner</h2>
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Configure Scenario</CardTitle>
          </CardHeader>
          <CardContent>
            <Button onClick={runScenario}>Run Scenario</Button>
          </CardContent>
        </Card>
        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Results</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Ridership change: {result.changes.ridership}%</p>
              <p>Wait time change: {result.changes.avg_wait}%</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
```

**Step 2:** Commit.

```bash
git add dashboard/src/routes/ScenarioPlanner.tsx
git commit -m "feat: add scenario planner frontend UI"
```

---

## Sub-Project 5: Streamlit Research UI Enhancement

### Task 5.1: Add real data toggle and GeoJSON export

**Files:**
- Modify: `main.py`

**Step 1:** Add real data toggle to Streamlit sidebar.

Find the Streamlit UI setup and add:

```python
st.sidebar.title("Data Source")
use_real_data = st.sidebar.checkbox("Use real OSM data", value=False, help="Toggle between synthetic and real Astana bus network from OpenStreetMap")

if use_real_data:
    st.sidebar.info("Loading real OSM data... This may take a moment.")
```

**Step 2:** Add GeoJSON export function.

```python
def export_geojson(network: NetworkSpec) -> str:
    """Export network as GeoJSON FeatureCollection."""
    import json
    features = []
    for i, name in enumerate(network.station_names):
        # Note: real data has lat/lon, synthetic does not — this is a placeholder
        features.append({
            "type": "Feature",
            "properties": {"name": name, "district": network.station_district[i]},
            "geometry": {"type": "Point", "coordinates": [71.4, 51.15]},  # placeholder
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
```

**Step 3:** Add export button to Streamlit.

```python
if st.sidebar.button("Export GeoJSON"):
    geojson = export_geojson(network)
    st.sidebar.download_button("Download GeoJSON", geojson, "astana_network.geojson")
```

**Step 4:** Commit.

```bash
git add main.py
git commit -m "feat: add real data toggle and GeoJSON export to Streamlit"
```

---

### Task 5.2: Add model comparison view

**Files:**
- Modify: `main.py`

**Step 1:** Add a model comparison tab in Streamlit.

```python
with st.expander("Model Comparison"):
    st.markdown("Compare DTS-GSSF against baseline models.")
    comparison_data = {
        "Model": ["DTS-GSSF", "LSTM", "GRU", "TCN", "Seasonal Naive"],
        "MAE": [6.38, 7.28, 7.15, 7.02, 8.42],
        "RMSE": [9.76, 11.05, 10.92, 10.78, 12.85],
    }
    st.bar_chart(comparison_data, x="Model", y=["MAE", "RMSE"])
```

**Step 2:** Commit.

```bash
git add main.py
git commit -m "feat: add model comparison view to Streamlit"
```

---

## Sub-Project 6: Real-Time Data Pipeline & Deployment

### Task 6.1: Create docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

**Step 1:** Write Docker Compose configuration.

```yaml
version: "3.8"

services:
  postgres:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: michi
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/michi
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  dashboard:
    build: ./dashboard
    ports:
      - "80:80"
    depends_on:
      - backend

  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://backend:8000
    depends_on:
      - backend

  celery:
    build: ./backend
    command: celery -A backend.tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/michi
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres

  celery-beat:
    build: ./backend
    command: celery -A backend.tasks beat --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/michi
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres

volumes:
  pgdata:
```

**Step 2:** Commit.

```bash
git add docker-compose.yml
git commit -m "feat: add Docker Compose for full stack deployment"
```

---

### Task 6.2: Create root Dockerfile for Streamlit

**Files:**
- Create: `Dockerfile`

**Step 1:** Write Dockerfile.

```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY main.py model_evaluation.py generate_figures.py ./
COPY data/ ./data/
COPY MODEL_ARCHITECTURE.md README.md ./

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Step 2:** Commit.

```bash
git add Dockerfile
git commit -m "feat: add root Dockerfile for Streamlit service"
```

---

### Task 6.3: Seed database script

**Files:**
- Create: `backend/seed.py`

**Step 1:** Write seed script.

```python
"""Seed database with sample stations and routes."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database import Base, engine
from backend.models import Station, Route  # SQLAlchemy models


def seed():
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        # Add sample stations
        stations = [
            Station(stop_id="stop_1", name="Khan Shatyr", lat=51.1605, lon=71.4702, district="Esil"),
            Station(stop_id="stop_2", name="Mega Silk Way", lat=51.1450, lon=71.4300, district="Almaty"),
            Station(stop_id="stop_3", name="Nurzhol Blvd", lat=51.1500, lon=71.4500, district="Saryarka"),
        ]
        session.add_all(stations)
        session.commit()
        print("Seeded database with sample data")


if __name__ == "__main__":
    seed()
```

**Step 2:** Commit.

```bash
git add backend/seed.py
git commit -m "feat: add database seed script"
```

---

### Task 6.4: Update README

**Files:**
- Modify: `README.md`

**Step 1:** Add deployment instructions.

```markdown
## Deployment

### Docker Compose (Full Stack)

```bash
docker compose up --build
```

Services:
- Dashboard: http://localhost
- Streamlit: http://localhost:8501
- API Docs: http://localhost:8000/docs
- Backend: http://localhost:8000

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | postgresql://postgres:postgres@localhost:5432/michi | PostgreSQL connection |
| REDIS_URL | redis://localhost:6379/0 | Redis connection |
| VITE_API_URL | http://localhost:8000/api/v1 | Frontend API base URL |
```

**Step 2:** Commit.

```bash
git add README.md
git commit -m "docs: update README with deployment instructions"
```

---

## Execution Order

1. Sub-Project 1 (Real Data Foundation) → Tasks 1.1–1.5
2. Sub-Project 2 (Backend & API) → Tasks 2.1–2.5
3. Sub-Project 3 (Dashboard) → Tasks 3.1–3.7
4. Sub-Project 4 (Scenario Simulator) → Tasks 4.1–4.2
5. Sub-Project 5 (Streamlit Enhancement) → Tasks 5.1–5.2
6. Sub-Project 6 (Deployment) → Tasks 6.1–6.4

**Parallelizable:** Sub-Project 2 and 3 can be developed in parallel after Sub-Project 1. Sub-Project 4 depends on both 2 and 3. Sub-Project 5 is independent. Sub-Project 6 comes last.

---

*End of Implementation Plan*
