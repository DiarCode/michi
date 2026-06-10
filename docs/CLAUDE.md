# Michi — Architecture Overview

## What It Is

A production-grade transit intelligence platform for the Astana bus network. Combines a deep-learning model (DTS-GSSF) with real-time operational dashboards.

## Stack

| Layer | Tech | Port |
|-------|------|------|
| Backend API | FastAPI + Uvicorn | 8100 |
| Dashboard | React 19 + Vite + TypeScript + Zustand | 3100 (nginx) / 5173 (dev) |
| Research UI | Streamlit + Plotly | 8600 |
| DB | SQLite (dev) / TimescaleDB (prod) | — |
| Cache/Broker | Redis 7 | 6380 |
| Tasks | Celery worker + beat | — |
| Model | PyTorch DTS-GSSF (3277-line `main.py`) | — |

## Key Files

```
main.py                    # Monolithic ML pipeline (data gen → train → online sim → Streamlit UI)
backend/
  app.py                   # FastAPI entry, router registration, lifespan
  database.py              # SQLAlchemy engine + session factory
  models_orm.py            # 11 ORM models (Station, Route, Alert, Ridership, Forecast, etc.)
  models.py                # Pydantic response schemas
  seed.py                  # DB seed script
  websocket.py             # WS manager + Redis pub/sub relay
  tasks.py                 # Celery task definitions
  routers/                 # 10 route modules (stations, routes, alerts, analytics, etc.)
  services/                # Business logic (forecast, alert, simulation, realtime, etc.)
  ml/                      # Model loading, prediction, drift detection, Kalman filter
dashboard/src/
  App.tsx                  # Root app with role-based nav
  routes/                  # 14 page components (LiveMap, Alerts, Scenarios, Forecast, etc.)
  stores/                  # 4 Zustand stores (simulation, bus, connection, timeline)
  hooks/                   # 5 custom hooks (WebSocket, alerts, forecasts, stations, timeline)
  lib/api.ts               # Axios API client (all endpoints)
  components/map/          # MapContainer, BusMarker, StationMarker, TimelineBar
  components/dashboard/    # KPIGrid, ForecastChart, CongestionHeatmap, SimulationMetrics, AlertTicker
data/
  osm_parser.py            # OSM data fetcher for Astana bus network
  generate_network.py      # Synthetic network generator
  generate_historical.py   # Historical ridership generator
experiments/               # Multi-seed evaluation, baselines, calibration, attribution
paper/                     # LaTeX thesis (memoirthesis.tex + chapters)
alembic/                   # DB migrations (3 versions)
```

## Request Lifecycle

1. Dashboard fetches via Axios → `/api/v1/*` (FastAPI)
2. FastAPI router → service layer → SQLAlchemy ORM → SQLite
3. Forecast service tries cached DTS-GSSF predictions, falls back to mock
4. WebSocket `/ws/realtime` pushes bus positions (5s) + Redis simulation events
5. Celery tasks: simulation engine, drift checks, alert generation

## Running Locally

```bash
# Backend
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8100

# Dashboard
cd dashboard && npm install && npm run dev

# Full stack
docker-compose up --build
```

## Tests

```bash
pytest tests/                    # Integration tests (API, ORM, services)
pytest backend/tests/             # ML model tests
```

## Known Issues

- See `TO_FIX.md` for Q1-critical thesis issues (C1–C6, H1–H12)
- `main.py` is 3277 lines — monolithic Streamlit + ML pipeline, needs decomposition
- Real-time bus positions are random-walk simulated (no real GPS feed)
- WebSocket has no auth, CORS is `*`
- No test coverage for ML pipeline or frontend