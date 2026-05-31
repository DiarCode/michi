# Michi — Astana Passenger Flow Forecasting Platform

A production-grade platform for forecasting passenger flow across Astana's public bus network, combining a deep-learning DTS-GSSF model with real-time operational dashboards.

## Project Overview

Michi provides end-to-end transit intelligence for the Astana bus network:

- **Prediction engine**: DTS-GSSF (Dual-Timescale Graph State-Space Forecasting) with Kalman residual correction, drift detection, and hierarchical reconciliation
- **Operational dashboard**: Role-based React SPA with live bus tracking, alert management, scenario planning, and forecast visualization
- **Research UI**: Streamlit-based interface for model training, comparison, and analytics
- **Real-time data**: WebSocket-backed bus position streaming and simulation event relay

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  Dashboard   │────▶│  FastAPI     │────▶│  SQLite /     │
│  React 19    │◀───│  Backend     │◀───▶│  TimescaleDB   │
│  :3100       │ ws  │  :8100      │     └────────────────┘
└─────────────┘     ├──────────────┤     ┌────────────────┐
                    │  WebSocket   │────▶│  Redis :6380   │
┌─────────────┐     │  /ws/realtime │     └────────────────┘
│  Streamlit   │────▶│              │     ┌────────────────┐
│  Research UI │     ├──────────────┤     │  Celery worker │
│  :8600       │     │  Celery beat │────▶│  + simulation  │
└─────────────┘     └──────────────┘     └────────────────┘
```

| Component | Technology | Port | Role |
|-----------|-----------|------|------|
| Backend API | FastAPI + Uvicorn | 8100 | REST API, WebSocket bus/simulation stream |
| Dashboard | React 19 + Vite + TypeScript | 3100 | Operational dashboards with 6 role views |
| Streamlit | Streamlit + Plotly | 8600 | Model training, analytics, comparison |
| Redis | Redis 7 Alpine | 6380 | Celery broker + WebSocket pub/sub |
| Celery worker | Celery + Redis | — | Background tasks, simulation engine |
| Celery beat | Celery + Redis | — | Scheduled tasks (alert generation, drift checks) |

## Quick Start

### Prerequisites

- Python 3.13+
- Node.js 20+
- Docker + Docker Compose

### Full Stack (Docker Compose)

```bash
# Clone and start all services
git clone <repo-url> && cd michi
docker-compose up --build
```

This starts all 6 services. Access:

- **Dashboard**: http://localhost:3100
- **Backend API docs**: http://localhost:8100/docs
- **Streamlit research UI**: http://localhost:8600
- **Health check**: http://localhost:8100/health

### Local Development (Backend only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8100
```

### Local Development (Dashboard only)

```bash
cd dashboard
npm install
npm run dev    # Vite dev server on :5173
```

Point the dashboard at a running backend by setting `VITE_API_URL` (defaults to `http://localhost:8100`).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./data/michi.db` | Database connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for Celery broker and pub/sub |
| `VITE_API_URL` | `http://localhost:8100` | Backend URL for the dashboard client |
| `API_URL` | `http://backend:8000` | Backend URL for Streamlit (Docker internal) |
| `CELERY_BROKER_URL` | (falls back to `REDIS_URL`) | Explicit Celery broker override |
| `MODEL_ARTIFACT_PATH` | `checkpoints/model_best.pt` | Path to DTS-GSSF model weights |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/stations` | List all bus stations |
| GET | `/api/v1/stations/{id}/detail` | Station detail with forecast and alerts |
| GET | `/api/v1/stations/{id}/forecast` | 24h forecast for a station |
| GET | `/api/v1/routes` | List all bus routes |
| GET | `/api/v1/routes/{id}/stops` | Stops on a route |
| GET | `/api/v1/routes/{id}/forecast` | Route-level forecast |
| GET | `/api/v1/routes/{id}/schedule` | Timetable for a route |
| GET | `/api/v1/dashboard/kpis` | Real-time KPI snapshot |
| GET | `/api/v1/dashboard/operations` | Operations report |
| GET | `/api/v1/dashboard/suggestions` | Optimization suggestions |
| GET | `/api/v1/alerts` | List all alerts |
| GET | `/api/v1/alerts/active` | Active (unacknowledged) alerts |
| GET | `/api/v1/alerts/rich` | Enriched alerts with context |
| POST | `/api/v1/alerts/{id}/ack` | Acknowledge an alert |
| POST | `/api/v1/scenarios/run` | Run what-if scenario |
| GET | `/api/v1/analytics/summary` | Ridership analytics by district |
| GET | `/api/v1/analytics/trends` | Trend data over time period |
| GET | `/api/v1/analytics/graph` | Network graph structure |
| GET | `/api/v1/analytics/predictions` | Multi-horizon predictions |
| GET | `/api/v1/interventions` | List interventions |
| POST | `/api/v1/interventions` | Create intervention |
| GET | `/api/v1/executive/kpis` | Executive dashboard KPIs |
| GET | `/api/v1/depot/status` | Depot bus availability |
| GET | `/api/v1/passenger/crowding` | Passenger crowding levels |
| POST | `/api/v1/simulation/start` | Start simulation |
| POST | `/api/v1/simulation/stop` | Stop simulation |
| GET | `/api/v1/simulation/state` | Current simulation state |
| GET | `/api/v1/timeline` | Timeline data for replay |
| GET | `/api/v1/simulation/metrics` | Historical MAE/MAPE time series |
| WS | `/ws/realtime` | Live bus positions + simulation events |
| GET | `/health` | Health check |

## Project Structure

```
michi/
├── backend/               # FastAPI application
│   ├── app.py             # Entry point, router registration
│   ├── database.py        # DB session + initialization
│   ├── models.py          # Pydantic response models
│   ├── models_orm.py      # SQLAlchemy ORM models
│   ├── websocket.py       # WebSocket bus/sim stream
│   ├── tasks.py           # Celery task definitions
│   ├── routers/           # API route handlers
│   ├── services/          # Business logic layer
│   ├── ml/                # DTS-GSSF model code
│   └── seed.py            # Database seed script
├── dashboard/             # React operational dashboard
│   ├── src/
│   │   ├── App.tsx        # Root app with role-based nav
│   │   ├── routes/        # Page components
│   │   ├── components/    # Shared UI components
│   │   ├── hooks/         # Custom React hooks
│   │   ├── stores/        # Zustand state stores
│   │   ├── lib/           # API client, constants
│   │   └── types/         # TypeScript interfaces
│   └── package.json
├── streamlit_app.py        # Research UI
├── docker-compose.yml      # Full-stack deployment
├── main.py                 # Paper experiments entry
├── paper/                  # LaTeX thesis
└── checkpoints/            # Model artifacts
```

## Model

The DTS-GSSF model architecture:

- **Dual timescales**: High-frequency station-level + low-frequency aggregated forecasting
- **Graph state-space**: Physical bus network adjacency constrains latent state transitions
- **Kalman correction**: Online residual correction during inference
- **Drift detection**: Page-Hinkley test triggers LoRA adaptation when distribution shifts
- **Hierarchical reconciliation**: Bottom-up forecasts aggregated to line and district totals

## License

MIT