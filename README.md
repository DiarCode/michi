# Michi — Astana Passenger Flow Forecasting Platform

A production-grade platform for forecasting passenger flow across Astana's public bus network, combining a deep-learning DTS-GSSF model with real-time operational dashboards.

## Architecture

- **Model**: Dual-Timescale Graph State-Space Forecasting (DTS-GSSF) with Kalman residual correction, drift detection, and hierarchical reconciliation
- **Data**: Real OpenStreetMap bus stops and routes for Astana, with synthetic ridership generation
- **Backend**: FastAPI REST API + WebSocket for live bus tracking, PostgreSQL + TimescaleDB, Redis, Celery
- **Operational Dashboard**: React 19 + Vite + TypeScript + Tailwind CSS + TanStack Query + Recharts + Zustand
- **Research UI**: Streamlit with model training, analytics, live simulation, network graph, operational forecast, and model comparison
- **Deployment**: Docker Compose full-stack

## Quick Start

### Prerequisites
- Python 3.13+
- Node.js 20+ (for dashboard)
- Docker + Docker Compose (for full stack)

### Local Development (Streamlit only)

      Welcome to Streamlit!

      If you'd like to receive helpful onboarding emails, news, offers, promotions,
      and the occasional swag, please enter your email address below. Otherwise,
      leave this field blank.

      Email: [0m

### Full Stack (Docker Compose)
#1 [internal] load local bake definitions
#1 reading from stdin 2.38kB done
#1 DONE 0.0s

#2 [backend internal] load build definition from Dockerfile
#2 transferring dockerfile: 370B 0.0s done
#2 DONE 0.1s

#3 [backend internal] load metadata for docker.io/library/python:3.13-slim
#3 ...

#4 [streamlit internal] load build definition from Dockerfile
#4 transferring dockerfile: 466B 0.0s done
#4 DONE 0.1s

#5 [dashboard internal] load build definition from Dockerfile
#5 transferring dockerfile: 224B done
#5 DONE 0.1s

#3 [streamlit internal] load metadata for docker.io/library/python:3.13-slim
#3 DONE 1.6s

#6 [celery-beat internal] load .dockerignore
#6 transferring context: 2B done
#6 DONE 0.0s

#7 [streamlit internal] load .dockerignore
#7 transferring context: 2B done
#7 DONE 0.0s

#8 [celery-beat internal] load build context
#8 transferring context: 557B done
#8 DONE 0.0s

#9 [backend 2/6] WORKDIR /app
#9 CACHED

#10 [backend 3/6] RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
#10 CACHED

#11 [backend 4/6] COPY backend/requirements.txt .
#11 ERROR: failed to calculate checksum of ref cide9xwwm2lifgzixwqwi1tpc::4zhpsaer2q46d28qo6obwypkp: "/backend/requirements.txt": not found

#12 [streamlit 1/6] FROM docker.io/library/python:3.13-slim@sha256:b04b5d7233d2ad9c379e22ea8927cd1378cd15c60d4ef876c065b25ea8fb3bf3
#12 resolve docker.io/library/python:3.13-slim@sha256:b04b5d7233d2ad9c379e22ea8927cd1378cd15c60d4ef876c065b25ea8fb3bf3 0.0s done
#12 sha256:e4f4b0eb5b0dcc71159795f3bcefadea9394f9bd56105f769cb0a33119057250 0B / 250B 0.2s
#12 sha256:db744c23eac55bf8d5cd5fce3abb1457a52434c36e5743e1a330b650f28284e5 0B / 1.29MB 0.2s
#12 sha256:f3b83f2d9173c5c97086327efdd961d46c8dd661b391fa56c7445ca8c34f1660 0B / 11.82MB 0.2s
#12 sha256:5b4d6ff92fc4e14e911b7753c954fac965d48c40fe1075758d284148ccace970 0B / 29.78MB 0.2s
#12 DONE 0.3s

#13 [dashboard internal] load metadata for docker.io/library/node:20-alpine
#13 CANCELED

#14 [dashboard internal] load metadata for docker.io/library/nginx:alpine
#14 CANCELED

#9 [streamlit 2/6] WORKDIR /app
#9 CANCELED

#15 [streamlit internal] load build context
#15 transferring context: 30.50MB 0.3s done
#15 CANCELED
------
 > [backend 4/6] COPY backend/requirements.txt .:
------

Services:
-  — TimescaleDB on port 5432
-  — Cache and Celery broker on port 6379
-  — FastAPI on port 8000
-  — React app on port 80
-  — Research UI on port 8501
-  — Background task workers
-  — Scheduled task scheduler

## Project Structure



## API Endpoints

| Endpoint | Description |
|----------|-------------|
|  | List all bus stations |
|  | List all bus routes |
|  | KPI snapshot |
|  | Active alerts |
|  | Run what-if scenario |
|  | WebSocket live bus positions |

## Model

The DTS-GSSF model architecture:
- **Dual timescales**: High-frequency station-level + low-frequency aggregated forecasting
- **Graph state-space**: Physical bus network adjacency constrains latent state transitions
- **Kalman correction**: Online residual correction during inference
- **Drift detection**: Page-Hinkley test triggers LoRA adaptation when distribution shifts
- **Hierarchical reconciliation**: Bottom-up forecasts aggregated to line and district totals

## License

MIT
