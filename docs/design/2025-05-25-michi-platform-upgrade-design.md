# Michi Platform Upgrade — Design Specification

**Date:** 2025-05-25
**Status:** Approved for implementation
**Approach:** Hybrid (React Operational Dashboard + Enhanced Streamlit Research UI)
**Author:** Claude (Vertex)

---

## 1. Executive Summary

Transform the existing DTS-GSSF research prototype into a production-grade **Astana Transit Intelligence Platform**. The platform serves three stakeholder groups with distinct needs: city transport planners (scenario analysis, budget justification), bus operations dispatch managers (real-time command center, anomaly alerts), and data science researchers (model training, evaluation, ablation studies).

The upgrade follows **Approach B (Hybrid)**: a React-based operational dashboard handles real-time visualization, scenario planning, and role-based views; the existing Streamlit UI is enhanced as the dedicated research/modeling interface.

---

## 2. Guiding Principles

1. **Real data over synthetic.** The platform must operate on real Astana bus stop coordinates, route geometries, and (eventually) real-time GPS feeds. Synthetic data is a fallback only.
2. **Actionable intelligence over raw metrics.** Every chart, alert, and forecast must answer a specific operational question.
3. **Incremental deployability.** Each sub-project produces a working increment that can be deployed and validated independently.
4. **Minimal diff over rewrite.** Preserve the existing DTS-GSSF model, training pipeline, and evaluation framework. New code integrates, it does not replace.

---

## 3. Sub-Project Decomposition

| # | Sub-Project | Duration Estimate | Depends On | Produces |
|---|-------------|-------------------|------------|----------|
| 1 | **Real Data Foundation** | 2–3 days | — | Real Astana bus network with coordinates, OSM parser script |
| 2 | **Professional Backend & API** | 4–5 days | 1 | FastAPI server, PostgreSQL schema, TimescaleDB hypertables, REST/WebSocket API |
| 3 | **Operational Dashboard (React + MapCN)** | 5–7 days | 2 | React app with real-time map, alerts, KPIs, role-based routing |
| 4 | **Scenario Simulator** | 3–4 days | 2, 3 | What-if planning UI, modified DTS-GSSF inference endpoint |
| 5 | **Streamlit Research UI Enhancement** | 2–3 days | 1 | Polished research interface, model comparison, export tools |
| 6 | **Real-Time Data Pipeline & Deployment** | 3–4 days | 2, 3 | Docker Compose, egov.kz adapter, automated retraining, production deploy |

**Total Estimated Duration:** 19–26 days (sequential with some parallelization possible)

---

## 4. Sub-Project 1: Real Data Foundation

### 4.1 Goal
Replace the synthetic `build_astana_network()` with a real Astana bus network parsed from OpenStreetMap (OSM), with fallback to synthetic if parsing fails.

### 4.2 Data Source
**Primary:** Geofabrik Kazakhstan OSM extract (`.osm.pbf`, updated daily). Filtered to Astana bounding box.

**Bounding Box:**
- West: 71.25°E
- East: 71.65°E
- South: 50.95°N
- North: 51.25°N

**Secondary (fallback):** MDPI 2024 GTFS dataset (3 routes, real GPS-derived). Attempted if OSM yields insufficient data.

### 4.3 Data Flow

```
Geofabrik Kazakhstan OSM (.osm.pbf)
    │
    ▼
osmium-tool extract (Astana bbox)
    │
    ▼
osmnx / custom parser (route=bus relations)
    │
    ▼
┌──────────────────────────────────────────┐
│ Extracted artifacts:                     │
│   stops.json       — stop_id, name, lat, │
│                      lon, district         │
│   routes.json      — route_id, name,     │
│                      route_type, color     │
│   shapes.json      — route_id, sequence, │
│                      lat, lon             │
│   stop_sequences.json — route_id,        │
│                      ordered stop_ids      │
│   adjacency.json   — physical graph       │
│                      (stop-to-stop edges)  │
└──────────────────────────────────────────┘
    │
    ▼
NetworkSpec builder (real coords + names)
    │
    ▼
DTS-GSSF DataBundle generation
```

### 4.4 Implementation

**New files:**
- `data/osm_parser.py` — OSM parsing logic
- `data/download_osm.py` — Geofabrik download + caching
- `data/cache/` — Persisted parsed network JSON
- `tests/test_osm_parser.py` — Unit tests

**Modified files:**
- `main.py`: `build_astana_network()` gains `use_real_data: bool = False` parameter

**Parser Logic:**
1. Download Kazakhstan `.osm.pbf` from `https://download.geofabrik.de/asia/kazakhstan-latest.osm.pbf`
2. Cache at `data/cache/kazakhstan-latest.osm.pbf`
3. Extract Astana bounding box using `osmium extract`
4. Parse `relation[route=bus]` with `osmnx` or `osmium tags-filter`
5. For each route relation:
   - Extract ordered list of `node[highway=bus_stop]` members
   - Get lat/lon from node coordinates
   - Build route polyline from `way` members
6. Build adjacency matrix: `A_phys[i,j] = 1` if stop `i` and stop `j` are consecutive on any route
7. Assign districts via point-in-polygon (Esil, Almaty, Saryarka, Baikonur) using OSM boundary relations

### 4.5 Error Handling

| Scenario | Behavior |
|----------|----------|
| OSM download fails | Log warning, fallback to synthetic network |
| No bus routes in bbox | Log warning, fallback to synthetic |
| Stop missing coordinates | Skip stop, interpolate from neighbors if possible |
| Cache stale (>30 days) | Re-download on next parser invocation |

### 4.6 Success Criteria
- [ ] `build_astana_network(use_real_data=True)` returns 30+ stops with valid lat/lon
- [ ] All coordinates fall within Astana city bounds
- [ ] 5+ bus routes with ordered stop sequences
- [ ] Physical adjacency matrix built from real stop-to-stop connections
- [ ] MapCN can render all stops as points on real Astana map

---

## 5. Sub-Project 2: Professional Backend & API

### 5.1 Goal
Build a production-grade FastAPI backend that serves forecasts, manages data persistence, and integrates with the frontend.

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ REST Routers   │  │ WebSocket     │  │ Background   │         │
│  │                │  │ Handler       │  │ Tasks        │         │
│  │ /api/v1/       │  │ /ws/realtime  │  │ (Celery)     │         │
│  │   /forecasts   │  │               │  │              │         │
│  │   /stations    │  │               │  │              │         │
│  │   /routes      │  │               │  │              │         │
│  │   /alerts      │  │               │  │              │         │
│  │   /scenarios   │  │               │  │              │         │
│  └──────┬─────────┘  └──────┬────────┘  └──────┬────────┘         │
│         │                   │                   │                 │
│         └───────────────────┴───────────────────┘                 │
│                          │                                     │
│              ┌──────────────▼──────────────┐                       │
│              │    Service Layer            │                       │
│              │  - ForecastService          │                       │
│              │  - ScenarioService          │                       │
│              │  - AlertService             │                       │
│              │  - RealtimeService          │                       │
│              └──────────────┬──────────────┘                       │
│                             │                                     │
│         ┌───────────────────┼───────────────────┐                 │
│         ▼                   ▼                   ▼                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ PostgreSQL    │  │ TimescaleDB   │  │ Redis         │         │
│  │ (metadata,    │  │ (time-series  │  │ (cache,      │         │
│  │  users,       │  │  ridership,   │  │  pub/sub)     │         │
│  │  config)      │  │  forecasts)   │  │               │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Database Schema (PostgreSQL + TimescaleDB)

**stations table:**
```sql
CREATE TABLE stations (
    id SERIAL PRIMARY KEY,
    stop_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    lat DECIMAL(10, 8) NOT NULL,
    lon DECIMAL(11, 8) NOT NULL,
    district VARCHAR(64),
    route_ids INTEGER[],
    created_at TIMESTAMP DEFAULT NOW()
);
```

**routes table:**
```sql
CREATE TABLE routes (
    id SERIAL PRIMARY KEY,
    route_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    color VARCHAR(7),
    stop_sequence INTEGER[],
    created_at TIMESTAMP DEFAULT NOW()
);
```

**ridership (TimescaleDB hypertable):**
```sql
CREATE TABLE ridership (
    time TIMESTAMPTZ NOT NULL,
    station_id INTEGER REFERENCES stations(id),
    route_id INTEGER REFERENCES routes(id),
    passenger_count INTEGER,
    avg_wait_time_min DECIMAL(5,2),
    weather_condition VARCHAR(32),
    is_event_day BOOLEAN DEFAULT FALSE
);
SELECT create_hypertable('ridership', 'time');
```

**forecasts (TimescaleDB hypertable):**
```sql
CREATE TABLE forecasts (
    time TIMESTAMPTZ NOT NULL,
    station_id INTEGER REFERENCES stations(id),
    horizon_steps INTEGER,
    predicted_count DECIMAL(10,2),
    confidence_lower DECIMAL(10,2),
    confidence_upper DECIMAL(10,2),
    model_version VARCHAR(32)
);
SELECT create_hypertable('forecasts', 'time');
```

**alerts table:**
```sql
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(32) NOT NULL,  -- 'overcrowding', 'delay', 'drift'
    severity VARCHAR(16) NOT NULL,    -- 'low', 'medium', 'high', 'critical'
    station_id INTEGER REFERENCES stations(id),
    route_id INTEGER REFERENCES routes(id),
    message TEXT NOT NULL,
    details JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 5.4 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/stations` | List all stations with coordinates |
| GET | `/api/v1/stations/{id}/forecast` | Get forecast for a station |
| GET | `/api/v1/routes` | List all routes |
| GET | `/api/v1/routes/{id}/stops` | Get ordered stops for a route |
| GET | `/api/v1/dashboard/kpis` | Current network KPIs |
| POST | `/api/v1/scenarios/run` | Run what-if scenario |
| GET | `/api/v1/alerts` | List alerts (filter by severity) |
| POST | `/api/v1/alerts/{id}/ack` | Acknowledge alert |
| WS | `/ws/realtime` | WebSocket for live bus positions & alerts |

### 5.5 WebSocket Protocol

```json
// Server → Client: bus position update
{
  "type": "bus_position",
  "data": {
    "bus_id": "BUS-001",
    "route_id": "Route 12",
    "lat": 51.1605,
    "lon": 71.4702,
    "speed_kmh": 32,
    "next_stop": "Khan Shatyr",
    "eta_seconds": 180,
    "occupancy_percent": 78
  }
}

// Server → Client: new alert
{
  "type": "alert",
  "data": {
    "id": 42,
    "type": "overcrowding",
    "severity": "high",
    "station": "Mega Silk Way",
    "message": "Station overcrowding predicted in 15 min",
    "timestamp": "2025-05-25T14:30:00Z"
  }
}
```

### 5.6 New Files
- `backend/app.py` — FastAPI application entry point
- `backend/routers/` — API route handlers
- `backend/services/` — Business logic services
- `backend/models.py` — Pydantic request/response models
- `backend/database.py` — SQLAlchemy + TimescaleDB setup
- `backend/websocket.py` — WebSocket manager
- `backend/tasks.py` — Celery background tasks
- `backend/Dockerfile`
- `docker-compose.yml` — Full stack orchestration

### 5.7 Success Criteria
- [ ] All API endpoints return correct JSON with OpenAPI docs at `/docs`
- [ ] WebSocket streams mock real-time bus positions at 5s intervals
- [ ] TimescaleDB hypertables created and populated with sample data
- [ ] Docker Compose brings up full backend stack with one command

---

## 6. Sub-Project 3: Operational Dashboard (React + MapCN)

### 6.1 Goal
Build the command-center UI for dispatchers and planners. This is the face of the platform.

### 6.2 Tech Stack
- **Framework:** React 19 + Vite + TypeScript
- **UI Library:** shadcn/ui (Tailwind CSS + Radix primitives)
- **State Management:** TanStack Query (React Query) + Zustand
- **Charts:** Recharts (dashboard KPIs) + Plotly (time-series)
- **Map:** MapCN (iframe or JS SDK integration)
- **Real-Time:** WebSocket client with automatic reconnection
- **Routing:** React Router v7
- **Icons:** Lucide React
- **Build:** Vite (dev) + Docker (production)

### 6.3 Application Structure

```
dashboard/
├── src/
│   ├── main.tsx                    # Entry point
│   ├── App.tsx                     # Router + providers
│   ├── routes/                     # Page components
│   │   ├── CommandCenter.tsx       # Main dashboard (default)
│   │   ├── LiveMap.tsx             # Full-screen map view
│   │   ├── AlertsPage.tsx          # Alert history & management
│   │   ├── ScenarioPlanner.tsx     # What-if simulator UI
│   │   ├── Reports.tsx             # Scheduled reports
│   │   └── Settings.tsx            # User preferences
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx         # Navigation sidebar
│   │   │   ├── TopBar.tsx          # Header with role selector
│   │   │   └── RoleGuard.tsx       # Role-based access wrapper
│   │   ├── map/
│   │   │   ├── MapContainer.tsx    # MapCN wrapper
│   │   │   ├── BusMarker.tsx       # Animated bus icon
│   │   │   ├── StationMarker.tsx   # Station with congestion color
│   │   │   └── RoutePolyline.tsx   # Route overlay
│   │   ├── dashboard/
│   │   │   ├── KPIGrid.tsx         # Key metrics cards
│   │   │   ├── ForecastChart.tsx   # Ridership forecast chart
│   │   │   ├── CongestionHeatmap.tsx # Station load heatmap
│   │   │   └── AlertTicker.tsx     # Scrolling alert bar
│   │   ├── scenario/
│   │   │   ├── ScenarioForm.tsx    # Input controls
│   │   │   └── ScenarioResult.tsx  # Before/after comparison
│   │   └── ui/                     # shadcn/ui components
│   ├── hooks/
│   │   ├── useWebSocket.ts         # WebSocket connection hook
│   │   ├── useStations.ts          # TanStack Query for stations
│   │   ├── useForecasts.ts         # TanStack Query for forecasts
│   │   └── useAlerts.ts            # TanStack Query for alerts
│   ├── lib/
│   │   ├── api.ts                  # Axios client + interceptors
│   │   ├── websocket.ts            # WebSocket client class
│   │   └── utils.ts                # Date formatting, color maps
│   ├── types/
│   │   └── index.ts                # TypeScript interfaces
│   └── styles/
│       └── globals.css             # Tailwind entry
├── public/
│   └── mapcn-loader.html           # MapCN iframe fallback
├── index.html
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── package.json
└── Dockerfile
```

### 6.4 Role-Based Views

| Role | Default Page | Visible Tabs | Key Features |
|------|-------------|------------|--------------|
| **Dispatch Manager** | Command Center | Live Map, Alerts, KPIs | Real-time bus positions, delay warnings, crew reallocation suggestions |
| **City Planner** | Scenario Planner | Scenarios, Reports, KPIs | What-if analysis, trend reports, budget justification charts |
| **Executive / Mayor** | Reports | Reports, KPIs | High-level trends, YoY improvements, citizen satisfaction proxies |
| **Data Scientist** | (redirects to Streamlit) | — | Full model control, training, evaluation |

### 6.5 MapCN Integration

**Approach:** Use MapCN's JS SDK if available; otherwise iframe embedding.

```tsx
// MapContainer.tsx
<iframe
  src={`https://mapcn.dev/embed?center=51.1605,71.4702&zoom=13&markers=${markersParam}`}
  style={{ width: '100%', height: '100%', border: 'none' }}
  allow="geolocation"
/>
```

**Overlay Strategy:**
- MapCN displays the base map
- React overlays bus markers and route polylines using HTML5 Canvas or SVG layered on top of the iframe
- Alternative: Use MapCN's API to add custom markers if the SDK supports it

### 6.6 Real-Time Animation

- Bus markers update every 5 seconds via WebSocket
- Smooth interpolation between positions using CSS transitions
- Station congestion colors: green (0–60%), yellow (60–85%), red (85–100%)
- Clicking a station opens a detail panel with forecast chart and recent alerts

### 6.7 Success Criteria
- [ ] Map renders with all real Astana stations plotted
- [ ] Bus markers animate smoothly between positions
- [ ] Station colors reflect predicted congestion
- [ ] KPI grid updates in real-time
- [ ] Role selector changes visible UI elements
- [ ] Responsive design works on 1920x1080 and 1366x768

---

## 7. Sub-Project 4: Scenario Simulator

### 7.1 Goal
Enable transport planners to answer "what if" questions by modifying network parameters and re-running the DTS-GSSF model.

### 7.2 Supported Scenarios

| Scenario | Parameters | Output |
|----------|-----------|--------|
| **Frequency Change** | Route, new headway (min) | Impact on wait times, crowding |
| **Route Extension** | Add stops to a route | Ridership redistribution |
| **Station Closure** | Station, duration | Diverted passenger load |
| **Event Simulation** | Location, expected crowd | Surge predictions |
| **Fleet Reallocation** | Move N buses from route A to B | Service level changes |

### 7.3 Backend Implementation

The scenario engine creates a **modified DataBundle** and runs inference:

```python
@dataclass
class ScenarioConfig:
    base_config: DataGenConfig
    modifications: List[ScenarioMod]

@dataclass
class ScenarioMod:
    type: str  # 'frequency', 'closure', 'extension', 'event'
    target: str  # route_id or station_id
    params: Dict[str, Any]

def run_scenario(config: ScenarioConfig, model: DTSGSSF) -> ScenarioResult:
    # 1. Clone base DataBundle
    # 2. Apply modifications (e.g., zero out station ridership for closure)
    # 3. Recompute hierarchy (line/district totals)
    # 4. Run model inference on modified data
    # 5. Compare metrics: base vs scenario
    pass
```

**Endpoint:** `POST /api/v1/scenarios/run`

### 7.4 Frontend UI

- **Left panel:** Scenario builder with drag-and-drop modifications
- **Center:** Split-view map — base network on left, scenario network on right
- **Right panel:** Metric comparison cards (ridership, wait time, cost)
- **Bottom:** Time-series comparison chart (base vs scenario)

### 7.5 Success Criteria
- [ ] User can configure and run a scenario in <30 seconds
- [ ] Results show before/after comparison with % change
- [ ] Scenario can be saved and shared via URL
- [ ] Export to PDF report

---

## 8. Sub-Project 5: Streamlit Research UI Enhancement

### 8.1 Goal
Polish the existing Streamlit interface for data scientists while keeping it fully functional.

### 8.2 Improvements

| Area | Current State | Improvement |
|------|--------------|-------------|
| **Layout** | Basic tabs | Persistent sidebar navigation, collapsible sections |
| **Real Data Toggle** | Synthetic only | `use_real_data` checkbox that switches to OSM network |
| **Model Export** | Checkpoints only | Add ONNX export, TorchScript export, API model serving |
| **Data Export** | CSV only | Add Parquet, GeoJSON (stations), and GTFS output |
| **Comparison** | Single model | Side-by-side model comparison (DTS-GSSF vs baseline) |
| **Documentation** | None in UI | Inline help tooltips, methodology explanations |

### 8.3 New Features

1. **GeoJSON Export Tab:** Export the current network as GeoJSON for use in external GIS tools
2. **Model Registry:** Track multiple trained models with metadata (epochs, hyperparameters, metrics)
3. **Batch Forecast:** Upload a CSV of future dates, download predictions
4. **Drift Analysis:** Visualize drift scores over time with annotated events

### 8.4 Success Criteria
- [ ] Streamlit UI uses real OSM data when toggled
- [ ] All tabs have professional styling consistent with React dashboard
- [ ] Model comparison view shows 2+ models side-by-side
- [ ] Export functions produce valid GeoJSON and Parquet files

---

## 9. Sub-Project 6: Real-Time Data Pipeline & Deployment

### 9.1 Goal
Build the production deployment pipeline and integrate real-time data sources.

### 9.2 Docker Compose Stack

```yaml
services:
  postgres:
    image: timescale/timescaledb:latest-pg16
    volumes:
      - pgdata:/var/lib/postgresql/data
  redis:
    image: redis:7-alpine
  backend:
    build: ./backend
    depends_on: [postgres, redis]
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://redis:6379
  dashboard:
    build: ./dashboard
    ports: ["80:80"]
    depends_on: [backend]
  streamlit:
    build: .
    ports: ["8501:8501"]
    depends_on: [backend]
  celery:
    build: ./backend
    command: celery -A tasks worker --loglevel=info
    depends_on: [redis, postgres]
  celery-beat:
    build: ./backend
    command: celery -A tasks beat --loglevel=info
    depends_on: [redis, postgres]
```

### 9.3 Real-Time Data Ingestion (Future Phase)

**egov.kz Adapter:**
```python
class AstanaRealtimeAdapter:
    """Polls egov.kz real-time bus API and pushes to WebSocket + TimescaleDB."""
    
    async def poll(self):
        # 1. Fetch bus positions from egov.kz API
        # 2. Parse JSON into BusPosition objects
        # 3. Write to TimescaleDB (positions hypertable)
        # 4. Publish to Redis pub/sub
        # 5. WebSocket manager broadcasts to connected clients
        pass
```

**Schedule:** Polling every 10–30 seconds (respecting API rate limits)

### 9.4 Automated Model Retraining

**Celery Beat Schedule:**
```python
beat_schedule = {
    'retrain-model-weekly': {
        'task': 'tasks.retrain_model',
        'schedule': crontab(day_of_week=0, hour=2),  # Sundays at 2 AM
    },
    'generate-forecasts': {
        'task': 'tasks.generate_forecasts',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
    },
    'cleanup-old-data': {
        'task': 'tasks.cleanup_data',
        'schedule': crontab(day_of_month=1, hour=3),
    },
}
```

### 9.5 Success Criteria
- [ ] `docker compose up` starts all services
- [ ] Backend health check passes at `/health`
- [ ] WebSocket streams mock bus positions
- [ ] Celery tasks run on schedule (verified via logs)
- [ ] Frontend served on port 80, Streamlit on 8501

---

## 10. Data Flow Summary (End-to-End)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  OSM / Real-Time│     │  FastAPI        │     │  React Dashboard│
│  Data Sources   │────▶│  Backend        │────▶│  (Operational)  │
│                 │     │                 │     │                 │
│ - Geofabrik OSM │     │ - Model serving │     │ - MapCN map     │
│ - egov.kz API   │     │ - Scenario sim  │     │ - Bus tracking  │
│ - Weather APIs  │     │ - Alert engine  │     │ - Alerts        │
└─────────────────┘     │ - Forecast gen  │     └─────────────────┘
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  PostgreSQL     │
                        │  + TimescaleDB  │
                        │                 │
                        │ - Stations      │
                        │ - Routes        │
                        │ - Ridership     │
                        │ - Forecasts     │
                        │ - Alerts        │
                        └─────────────────┘
                                 ▲
                                 │
                        ┌─────────────────┐
                        │  Streamlit      │
                        │  (Research)     │
                        │                 │
                        │ - Model training│
                        │ - Evaluation    │
                        │ - Data export   │
                        └─────────────────┘
```

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OSM data incomplete for Astana | Medium | High | Fallback to synthetic + manual curation; integrate MDPI dataset if accessible |
| MapCN iframe integration limited | Medium | High | Evaluate MapCN JS SDK; fallback to Leaflet with custom tiles if needed |
| egov.kz API requires registration | High | Medium | Implement adapter architecture that can be enabled later; use mock data initially |
| Streamlit + React deployment complexity | Medium | Medium | Separate Docker services; nginx reverse proxy; clear documentation |
| Model inference latency too high for real-time | Low | High | Optimize with ONNX Runtime; cache forecasts; pre-compute common scenarios |
| Database performance with large time-series | Medium | Medium | TimescaleDB chunking; retention policies; materialized views for KPIs |

---

## 12. Open Questions (To Resolve During Implementation)

1. **MapCN SDK availability:** Does MapCN provide a JS SDK for custom markers, or only iframe embedding? This affects the map overlay strategy.
2. **egov.kz API authentication:** Does the real-time bus API require an API key, and what are the rate limits?
3. **User authentication:** Does the platform need user login/role management, or is role selection UI-only for now?

---

## 13. Appendix: File Structure (Post-Upgrade)

```
michi/
├── backend/                    # NEW: FastAPI backend
│   ├── app.py
│   ├── routers/
│   ├── services/
│   ├── models.py
│   ├── database.py
│   ├── websocket.py
│   ├── tasks.py
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/                  # NEW: React operational dashboard
│   ├── src/
│   ├── public/
│   ├── index.html
│   ├── vite.config.ts
│   ├── Dockerfile
│   └── package.json
├── data/                       # MODIFIED: Real data pipeline
│   ├── osm_parser.py           # NEW
│   ├── download_osm.py         # NEW
│   ├── cache/                  # NEW
│   └── ... (existing files)
├── main.py                     # MODIFIED: Real data integration
├── model_evaluation.py         # (existing)
├── generate_figures.py         # (existing)
├── pyproject.toml              # MODIFIED: Add backend dependencies
├── docker-compose.yml          # NEW
├── README.md                   # MODIFIED: Updated docs
└── docs/
    └── design/
        └── 2025-05-25-michi-platform-upgrade-design.md
```

---

*End of Design Specification*
