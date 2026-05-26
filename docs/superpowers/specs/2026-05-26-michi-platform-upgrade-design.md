# Michi Transit Intelligence Platform — Full Upgrade Design

**Date:** 2026-05-26
**Status:** Approved (pending user review)
**Architecture:** Hybrid Monolith + Celery Worker

---

## 1. Executive Summary

Transform Michi from a mock-data prototype into a production-grade transit intelligence platform serving 6 user roles. The DTS-GSSF model (currently isolated in `main.py`) will be extracted, integrated with the FastAPI backend, trained on realistic historical data, and serve real-time multi-horizon predictions. All frontend screens will be rebuilt with role-based access, real data, and actionable intervention workflows.

**Core business problems solved:**
1. Predict near-future passenger flow (15/30/60/120 min horizons) by route, trip, stop, corridor, zone
2. Detect abnormal demand early (weather, events, school peaks, disruptions)
3. Support intervention decisions (dispatch, headway, short-turn, reallocation, passenger info)
4. Create feedback loop between operations and model research for continuous improvement

---

## 2. Architecture

### 2.1 Hybrid Monolith + Celery

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Frontend   │  │   Backend    │  │  Celery      │   │
│  │   React+Vite │  │   FastAPI    │  │  Worker      │   │
│  │   :3100      │◄►│   :8100      │  │  (ML tasks)  │   │
│  └──────────────┘  └──────┬───────┘  └──────┬───────┘   │
│                          │                   │           │
│                    ┌─────┴─────┐      ┌──────┴──────┐   │
│                    │   Redis   │      │  Celery Beat │   │
│                    │   :6380   │      │  (scheduler) │   │
│                    └───────────┘      └─────────────┘   │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐                      │
│  │  Streamlit   │  │  SQLite/     │                      │
│  │  :8600       │  │  PostgreSQL  │                      │
│  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

**Key principle:** Celery workers handle compute-heavy ML tasks (training, backtesting, drift detection). FastAPI handles all API routes, serves predictions from cached model artifacts, and streams real-time data via WebSocket.

### 2.2 Backend Module Structure

```
backend/
├── app.py                    # FastAPI entry, lifespan, CORS
├── database.py               # SQLAlchemy engine + session
├── models_orm.py             # ORM models (expanded)
├── seed.py                   # Seed with OSM + 2GIS resolved data
├── ml/                       # DTS-GSSF extracted package
│   ├── __init__.py
│   ├── model.py              # GraphSSM architecture
│   ├── trainer.py             # Training loop + Celery tasks
│   ├── predictor.py           # Inference engine
│   ├── data_loader.py         # DB → feature tensors
│   ├── drift_detector.py      # Page-Hinkley + LoRA triggers
│   ├── kalman_filter.py       # Online residual correction
│   ├── hierarchical.py        # MinT/OLS reconciliation
│   └── artifact_store.py      # Model versioning + file management
├── routers/
│   ├── stations.py
│   ├── routes.py
│   ├── dashboard.py
│   ├── alerts.py
│   ├── scenarios.py
│   ├── analytics.py
│   ├── interventions.py       # NEW: intervention workflow
│   ├── depot.py               # NEW: depot operations
│   ├── passenger_info.py     # NEW: passenger-facing API
│   └── executive.py           # NEW: executive KPIs
├── services/
│   ├── alert_service.py       # Expanded: 7 alert families
│   ├── forecast_service.py    # Real DTS-GSSF predictions
│   ├── realtime_service.py    # Real-time bus tracking
│   ├── scenario_service.py
│   ├── intervention_service.py # NEW
│   ├── accuracy_service.py    # NEW: prediction accuracy tracking
│   └── suggestion_service.py  # NEW: optimization suggestions
├── tasks.py                   # Celery task definitions
└── websocket.py              # Real-time broadcast
```

---

## 3. Data Architecture

### 3.1 New Database Tables

```sql
-- Historical ridership (core training data)
CREATE TABLE historical_ridership (
    id INTEGER PRIMARY KEY,
    station_id VARCHAR(20) REFERENCES stations(stop_id),
    route_id VARCHAR(20) REFERENCES routes(route_id),
    timestamp DATETIME NOT NULL,
    passengers_boarding INTEGER NOT NULL,
    passengers_alighting INTEGER NOT NULL,
    load INTEGER NOT NULL,
    weather_code VARCHAR(10),
    temperature FLOAT,
    is_holiday BOOLEAN DEFAULT FALSE,
    is_event_day BOOLEAN DEFAULT FALSE,
    day_of_week INTEGER,
    hour INTEGER
);

-- Weather readings
CREATE TABLE weather_readings (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    temperature FLOAT,
    precipitation FLOAT,
    wind_speed FLOAT,
    visibility FLOAT,
    weather_code VARCHAR(10),
    sudden_change BOOLEAN DEFAULT FALSE
);

-- Events calendar
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    name VARCHAR(300) NOT NULL,
    venue VARCHAR(200),
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    expected_attendance INTEGER,
    affected_routes TEXT,   -- JSON array of route_ids
    affected_stations TEXT, -- JSON array of stop_ids
    event_type VARCHAR(50)  -- concert, sports, fair, ceremony, etc.
);

-- Interventions
CREATE TABLE interventions (
    id INTEGER PRIMARY KEY,
    alert_id INTEGER REFERENCES alerts(id),
    intervention_type VARCHAR(50) NOT NULL,  -- dispatch, short_turn, hold, deadhead, passenger_info
    route_id VARCHAR(20) REFERENCES routes(route_id),
    station_id VARCHAR(20) REFERENCES stations(stop_id),
    created_at DATETIME NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, approved, executing, completed, cancelled
    operator_note TEXT,
    predicted_impact TEXT,   -- JSON: {ridership_change, wait_time_change}
    actual_impact TEXT,      -- JSON: same structure, filled post-evaluation
    approved_by VARCHAR(100)
);

-- Model artifacts
CREATE TABLE model_artifacts (
    id INTEGER PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    artifact_path VARCHAR(500) NOT NULL,
    metrics_json TEXT,       -- JSON: {mae, rmse, mape, mase}
    training_config_json TEXT, -- JSON: {epochs, learning_rate, features, ...}
    dataset_hash VARCHAR(64),
    feature_version INTEGER DEFAULT 1,
    created_at DATETIME NOT NULL,
    is_production BOOLEAN DEFAULT FALSE,
    is_shadow BOOLEAN DEFAULT FALSE
);

-- Prediction accuracy tracking
CREATE TABLE prediction_accuracy (
    id INTEGER PRIMARY KEY,
    model_version VARCHAR(50) REFERENCES model_artifacts(version),
    station_id VARCHAR(20) REFERENCES stations(stop_id),
    route_id VARCHAR(20) REFERENCES routes(route_id),
    forecast_timestamp DATETIME NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    predicted FLOAT NOT NULL,
    actual FLOAT,
    absolute_error FLOAT,
    mape FLOAT,
    evaluated_at DATETIME
);

-- Extend forecasts table with horizon
ALTER TABLE forecasts ADD COLUMN horizon_minutes INTEGER DEFAULT 60;
ALTER TABLE forecasts ADD COLUMN route_id VARCHAR(20);
```

### 3.2 Historical Data Generator

**Script:** `data/generate_historical.py`

Generates 365 days x 24 hours of realistic Astana ridership data with:

| Factor | Implementation |
|--------|---------------|
| Time-of-day | Sinusoidal with morning peak (7-9), evening peak (17-19), overnight low |
| Day-of-week | Weekday factor 1.0, Saturday 0.7, Sunday 0.5 |
| Seasonal | Winter (Nov-Mar) +15% demand, Summer (Jun-Aug) -10% |
| Weather | Cold/snow increases demand +10-25%, rain +5-15% |
| Events | Astana Arena events: +200% nearby stations for 2h around event |
| Holidays | Reduced to weekend pattern |
| Route topology | Boarding peaks at early stops, alighting peaks at terminal stops |
| Random noise | Gaussian noise +/-10-15% |
| Anomalies | 2% of records get extreme spikes (sensor failure, sudden surge) |

**Output:** Populates `historical_ridership`, `weather_readings`, and `events` tables.

**Volume target:** 200+ stations x 365 days x 24 hours = ~1.75M rows. Sufficient for DTS-GSSF training.

### 3.3 District Resolution

For stations with `district = "Unknown"`:
1. Use 2GIS reverse geocoding API (free tier) to resolve lat/lon to Astana district
2. Fallback: OSM Nominatim API
3. Astana districts: Esil (left bank), Almaty (old center), Saryarka (right bank), Baikonur (industrial)
4. Update seed data with resolved districts

---

## 4. DTS-GSSF Model Integration

### 4.1 Model Extraction

Extract the complete DTS-GSSF implementation from `main.py` into `backend/ml/`:

- **`model.py`** — GraphSSM backbone: graph-structured state-space layers, multi-head spatial attention, temporal convolution
- **`trainer.py`** — Training loop with: AdamW optimizer, cosine LR schedule, gradient clipping, early stopping, checkpoint saving, Celery task wrapper
- **`predictor.py`** — Inference: load artifact, build feature tensor from latest DB data, run forward pass, produce 4-horizon predictions with confidence intervals
- **`data_loader.py`** — Query `historical_ridership` + `weather_readings` + `events`, build sliding-window feature tensors with proper normalization
- **`drift_detector.py`** — Page-Hinkley test on prediction residuals, trigger LoRA-style low-rank adaptation when drift exceeds threshold
- **`kalman_filter.py`** — Online residual correction: fast-timescale Kalman filter updates state estimate from latest observations
- **`hierarchical.py`** — MinT/OLS weighted projection reconciliation: station predictions to route to district to city aggregate
- **`artifact_store.py`** — Save/load PyTorch model checkpoints, version management, production/shadow flag

### 4.2 Celery Tasks

```python
# backend/tasks.py

@celery_app.task(bind=True)
def train_dts_gssf(self, config: dict):
    """Full training: load data, train model, save artifact, evaluate, set as production."""
    ...

@celery_app.task
def run_backtest(model_version: str, date_range: tuple):
    """Replay historical period with model, compute per-route/per-stop metrics."""
    ...

@celery_app.task
def detect_drift():
    """Check Page-Hinkley on recent prediction errors. Trigger re-training if needed."""
    ...

@celery_app.task
def shadow_inference(model_version: str):
    """Run challenger model predictions alongside production, store for comparison."""
    ...

@celery_app.task
def generate_predictions():
    """Every 5 min: load production model, run inference for all stations, store results."""
    ...

@celery_app.task
def evaluate_prediction_accuracy():
    """Compare past predictions with now-observed actuals, store accuracy metrics."""
    ...

@celery_app.task
def evaluate_alert_conditions():
    """Every 2 min: check all alert rules against latest predictions + real-time data."""
    ...
```

### 4.3 Real-Time Prediction Pipeline

```
Real-time bus data → WebSocket broadcast → Frontend (map, panels)
                          |
                    Store in ridership table
                          |
           Every 5 min: Celery generate_predictions()
                          |
           Load production model artifact
                          |
           Build feature tensor from recent data
                          |
           Run DTS-GSSF forward pass (4 horizons)
                          |
           Store predictions in forecasts table
                          |
           Broadcast via WebSocket to all connected clients
                          |
           Every 15 min: evaluate_prediction_accuracy()
           (compare predictions from X min ago with actuals now)
```

### 4.4 Feedback Loop

```
Observation → Prediction Error → Drift Detection → LoRA Adaptation → Re-training
     ^                                                            |
     +---------- Improved Model <-- Retrained on expanded data ---+
```

- Prediction accuracy tracked per horizon, per route, per station, per time regime
- Drift detection triggers automatic re-training when residual distribution shifts
- Champion/challenger: new models run in shadow mode before production promotion
- Acceptance criteria: route-level MAPE < threshold, stop-level MAPE < threshold

---

## 5. Role-Based UI Architecture

### 5.1 Role System

```typescript
type UserRole = 
  | 'dispatch'      // Daily Operations / Dispatch Center
  | 'research'      // Research Engineers / Data Scientists
  | 'planning'      // Service Planning / Scheduling
  | 'executive'     // Executives / Transport Authority
  | 'depot'         // Depot Managers / Fleet Controllers
  | 'passenger';    // Passenger Information / Communications

const ROLE_NAV: Record<UserRole, NavItem[]> = {
  dispatch: [
    { to: '/', label: 'Operations Home', icon: BarChart },
    { to: '/map', label: 'Live Map', icon: Map },
    { to: '/route-command', label: 'Route Command', icon: Route },
    { to: '/stop-hub', label: 'Stop / Hub', icon: MapPin },
    { to: '/alerts', label: 'Alerts', icon: AlertTriangle },
  ],
  research: [
    { to: '/research', label: 'Research Lab', icon: FlaskConical },
    { to: '/training', label: 'Training', icon: BrainCircuit },
    { to: '/compare', label: 'Compare', icon: GitCompare },
    { to: '/analytics', label: 'Analytics', icon: TrendingUp },
  ],
  planning: [
    { to: '/planning', label: 'Planning Studio', icon: LayoutGrid },
    { to: '/scenarios', label: 'Scenario Planner', icon: FlaskConical },
    { to: '/analytics', label: 'Analytics', icon: TrendingUp },
    { to: '/reports', label: 'Reports', icon: FileText },
  ],
  executive: [
    { to: '/executive', label: 'Executive Dashboard', icon: BarChart3 },
    { to: '/reports', label: 'Reports', icon: FileText },
  ],
  depot: [
    { to: '/depot', label: 'Depot Operations', icon: Truck },
    { to: '/alerts', label: 'Alerts', icon: AlertTriangle },
  ],
  passenger: [
    { to: '/passenger', label: 'Passenger Info', icon: Users },
    { to: '/map', label: 'Live Map', icon: Map },
  ],
};
```

Role selector in sidebar header, stored in localStorage, dynamic nav rendering.

### 5.2 Screen Specifications

#### Screen 1: Operations Home (Command Center)
**Role:** Dispatch/Operations
**Purpose:** Immediate city-wide situational awareness

**Widgets:**
- Active alerts by severity (queue with SLA timers)
- Map of predicted crowding hotspots (next 30-60 min)
- Top 10 routes at risk (sorted by predicted overload probability)
- Weather/event badges affecting predictions
- Reserve fleet availability summary
- Optimization suggestions panel (actionable recommendations)

**Interactions:**
- Click route to drill into direction, stop sequence, next-hour demand curve
- Filter by depot, operator, corridor, event zone
- Acknowledge or assign alerts

#### Screen 2: Live Map
**Role:** Dispatch, Passenger Info
**Purpose:** Real-time city-wide view with prediction overlay

**Fixes and enhancements:**
- Light theme forced (theme="light" on Map component)
- Route filter: when route checked, fetch stops, render MapRoute polyline, highlight station markers
- Heatmap toggle: MapLibre GL heatmap layer weighted by station load/ridership
- Current time display: live clock + date + current conditions badge
- Multi-horizon prediction overlay: selector for 15/30/60/120 min, renders predicted demand as colored circles
- Station tooltips show current + predicted ridership
- Route paths shown with directional arrows
- District labels resolved (no more "Unknown")

#### Screen 3: Route Command View
**Role:** Dispatch
**Purpose:** Route-level intervention

**Widgets:**
- Actual vs predicted boardings/load by stop and trip (bar chart)
- Current bus positions on route map with headway gaps
- Risky stops ranked by overload probability
- Recommended interventions with predicted effect
- Intervention simulator: "What if I dispatch 1 more bus?"

**Interactions:**
- Simulate dispatching additional vehicle
- Compare predicted state with/without intervention
- Trigger action and log operator note

#### Screen 4: Stop / Hub View
**Role:** Dispatch
**Purpose:** Manage pressure at interchange hubs

**Widgets:**
- Predicted queue/crowding pressure gauge
- Arrivals/departures board with connecting routes
- Nearby event, venue, weather, road context
- Passenger messaging controls

#### Screen 5: Research Lab
**Role:** Research Engineers
**Purpose:** Model tuning and production trust

**Widgets:**
- Input freshness and missing-data chart
- Model performance by route and horizon
- Drift monitors (Page-Hinkley statistics, feature distribution shifts)
- Feature importance and local explanations (SHAP-like)
- Experiment registry (dataset version, feature version, model version, validation period)
- Time-sliced replay environment
- Champion vs challenger comparison

**Interactions:**
- Launch training experiments with configurable parameters
- Slice backtests by date, weather, event type, route cluster
- Replay a failure day minute by minute
- Approve/reject model deployment

#### Screen 6: Planning Studio
**Role:** Service Planning
**Purpose:** Structural decision-making

**Widgets:**
- Recurrent overload heatmaps (by stop, segment, hour, weekday, season, weather)
- Event impact library (historical impact of past events)
- Planned vs effective headway vs demand comparison
- Missed demand proxies and crowding exposure over time
- OD pressure approximation (from fare sequences)

**Interactions:**
- Build and save service scenarios
- Export route review packs
- Generate evidence for budget/planning committees

#### Screen 7: Executive Dashboard
**Role:** Executives
**Purpose:** Governance and ROI

**Widgets:**
- Overcrowding minutes prevented (trend)
- Demand forecast accuracy (trend)
- Headway stability and on-time reliability
- Intervention volume and success rate
- Most problematic corridors and depots
- Financial impact summary (fuel, labor, reserve bus use savings)
- Benchmark vs baseline periods

**Interactions:**
- Drill down into outliers only
- Generate board-level report (export)

#### Screen 8: Depot Operations
**Role:** Depot Managers
**Purpose:** Fleet readiness for intervention support

**Widgets:**
- Depot availability board (spare buses, fueling/charging, shift coverage)
- Predicted demand surges near depot service area
- Feasible dispatch recommendations constrained by available assets
- Technical restrictions and maintenance schedule

#### Screen 9: Passenger Info Portal
**Role:** Passenger Communications
**Purpose:** Turn operational knowledge into public messaging

**Widgets:**
- Predicted crowding and expected delay by route and stop
- Service changes approved by dispatch
- Event and weather alerts affecting passenger loads
- Communication templates tied to route alerts
- Multi-channel publish (app banner, web, operator channels)

---

## 6. Alerting Engine

### 6.1 Alert Families (7 types)

| Family | Trigger | Severity |
|--------|---------|----------|
| Overcrowding risk | Predicted load > 85% capacity within horizon | Warning to Critical at >95% |
| Supply-demand gap | Predicted demand > 1.5x current supply | Warning |
| Bunching amplification | Headway deviation > 2x planned headway | Warning |
| Event dispersal surge | Event end time within 30 min, affected routes | Info to Warning |
| Weather demand shift | Severe weather + predicted ridership change > 20% | Info to Warning |
| Prediction uncertainty | Model confidence < 70% for key routes | Info |
| Data quality | Missing data > 15% of expected inputs in last 15 min | Info to Warning |

### 6.2 Alert Structure

```typescript
interface RichAlert {
  id: number;
  family: AlertFamily;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  what: string;          // What is happening
  when: string;          // When expected
  where: string;         // Where it will happen
  why: string;           // Why the model believes this
  confidence: number;    // 0-1
  recommended_actions: ActionSuggestion[];
  consequence_if_ignored: string;
  sla_timer_minutes: number;
  acknowledged: boolean;
  assigned_to: string | null;
  station_id?: string;
  route_id?: string;
  created_at: string;
}
```

### 6.3 Generation Pipeline

1. Celery beat: `evaluate_alert_conditions` every 2 minutes
2. Load latest predictions for all stations/routes at all horizons
3. Load current real-time data (bus positions, occupancies)
4. Load weather data, event calendar
5. Evaluate each alert family's rules against current state
6. Create new alerts that do not duplicate existing un-acked alerts
7. Broadcast new alerts via WebSocket
8. Update SLA timers on existing alerts

---

## 7. Intervention Workflow

```
Alert Created → Acknowledged → Assigned to Operator
                                    |
                    Choose Intervention Type:
                    +-------------------------+
                    | Dispatch reserve bus     |
                    | Short-turn service       |
                    | Hold/release for bunching|
                    | Deadhead repositioning   |
                    | Passenger messaging      |
                    | Route reinforcement      |
                    +----------+--------------+
                               |
                    Impact Predictor runs:
                    "What-if" model simulation
                    Shows before/after forecast
                               |
                    +-----------+------------+
                    | Approve    | Override   |
                    | (accept    | (manual    |
                    | suggestion)| decision) |
                    +------+----+-------+-----+
                           |              |
                    Execute Intervention   Record reason
                           |
                    Monitor Outcome
                           |
                    Evaluate: actual vs predicted impact
                           |
                    Feed back to model improvement
```

---

## 8. Forecasting Enhancements

### 8.1 Multi-Horizon Time Selector

On ForecastPage and LiveMap:
- Radio group: "Now" | "+15 min" | "+30 min" | "+1 hour" | "+2 hours"
- Forecast chart shows full 24h prediction with "now" marker
- Selected horizon highlighted with confidence band
- Current actual ridership overlaid for comparison

### 8.2 Current-Time Anchoring

- All forecast displays anchor to current time
- "Now" line on all charts
- Live-updating as new real-time data arrives via WebSocket
- Auto-scroll forecast chart to keep "now" centered

### 8.3 Simulation Accuracy

- System continuously compares predictions with actuals
- Accuracy dashboard in Research Lab:
  - MAPE/RMSE by horizon (15/30/60/120 min)
  - MAPE by route, by station, by peak/off-peak, by weather regime
  - Calibration plot (predicted vs actual quantiles)
  - Accuracy trend over time (is the model improving?)

### 8.4 Champion/Challenger

- New model versions start in "shadow" mode
- Shadow predictions stored alongside production predictions
- Comparison dashboard shows side-by-side accuracy
- Promotion criteria: route-level MAPE < threshold, stop-level MAPE < threshold
- Manual promotion with approval workflow

---

## 9. Optimization Suggestions Engine

### 9.1 Rule-Based Suggestions

Based on current predictions and supply gaps, generate actionable recommendations:

| Condition | Suggestion |
|-----------|------------|
| Route predicted >85% load in 30 min | "Consider dispatching reserve bus to {route} — predicted {pct}% overload in 30 min" |
| Headway >2x planned on route | "Bunching detected on {route} — consider holding bus at {stop} for {min} min" |
| Event ending within 1 hour | "Pre-position {n} buses near {venue} for dispersal surge" |
| Weather downgrade + demand spike | "Severe weather expected — increase frequency on routes {routes} by {pct}%" |
| Station closure affecting transfers | "Reroute {route} via {alternative} to serve affected passengers" |
| Low demand route + high demand nearby | "Consider reallocation: move 1 bus from {low_route} to {high_route}" |

### 9.2 Model-Based Suggestions

The DTS-GSSF model can simulate intervention effects:
- For each active alert, compute the predicted impact of each possible intervention type
- Rank interventions by predicted ridership/wait-time improvement
- Show top 3 recommendations per alert with confidence scores

---

## 10. External Signals Hub

### 10.1 Weather Integration

- Daily weather forecast fetched from open weather API
- Stored in `weather_readings` table
- Features: temperature, precipitation, wind, visibility, sudden change indicators
- Used as model input features
- Displayed as overlay on operations screens

### 10.2 Event Calendar

- Pre-populated with known Astana events (Astana Arena, Expo, Khan Shatyr events)
- Stored in `events` table
- Each event has: venue, start/end time, expected attendance, affected routes/stations
- Event impact scoring: historical analysis of past event effects on ridership

### 10.3 Calendar Context

- Day-of-week, holiday calendar, school schedules
- Ramadan/seasonal effects
- Business district vs residential area patterns

---

## 11. Implementation Phases

### Phase 1: Data Foundation (Week 1)
- Expand station/route data with OSM + 2GIS district resolution
- Build historical data generator, populate 365 days of ridership
- Add all new DB tables (historical_ridership, weather, events, interventions, model_artifacts, prediction_accuracy)
- Migrate forecasts table (add horizon_minutes, route_id)

### Phase 2: DTS-GSSF Integration (Week 2)
- Extract DTS-GSSF from main.py into backend/ml/
- Implement Celery training task
- Implement prediction generation task (5-min cycle)
- Implement accuracy evaluation task
- Wire training page to real Celery tasks
- Wire forecast endpoints to real model predictions

### Phase 3: Core UI Fixes (Week 2-3)
- Fix live map: light theme, route filters, heatmap toggle, current time
- Add route path rendering (MapRoute polyline when route selected)
- Add multi-horizon prediction overlay on map
- Add district resolution (no more "Unknown")
- Add optimization suggestions panel

### Phase 4: Role System and Command Center (Week 3-4)
- Implement role selector + dynamic sidebar
- Rebuild Command Center as real-time ops tower
- Build Route Command View
- Build Stop/Hub View
- Expand alert engine (7 families, rich structure, SLA timers)

### Phase 5: Intervention Workflow (Week 4)
- Build intervention CRUD + status tracking
- Build impact predictor (what-if simulation)
- Build intervention approval/rejection flow
- Build post-shift outcome evaluation

### Phase 6: Research Lab and Planning (Week 5)
- Build Research Lab: experiment registry, model comparison, drift monitors, feature importance, backtest replay
- Build Planning Studio: overload heatmaps, seasonal impact, scenario planner, route review packs

### Phase 7: Executive, Depot, Passenger (Week 6)
- Build Executive Dashboard: KPI trends, ROI view, benchmark
- Build Depot Operations Board: fleet availability, dispatch recommendations
- Build Passenger Info Portal: crowding predictions, service changes, messaging

### Phase 8: Feedback Loop and Polish (Week 6-7)
- Wire prediction accuracy to drift detection to auto re-training
- Implement champion/challenger shadow mode
- End-to-end testing
- Performance optimization
- Documentation

---

## 12. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | SQLite for dev, PostgreSQL for production | SQLite works for single-node; PostgreSQL needed for concurrent writes |
| Model serving | Load artifact into memory on worker start | Low latency for predictions; re-load on model version change |
| Real-time updates | WebSocket broadcast from FastAPI | Already implemented, works well for <1000 concurrent clients |
| Prediction caching | Redis cache for latest predictions | Avoid DB hit every 5 min for all stations |
| Historical data volume | ~1.75M rows (1 year x 200 stations x 24h) | Manageable in SQLite, fine in PostgreSQL |
| Frontend state | React Query for server state, Zustand for client state | Already using React Query; add Zustand for role/UI state |
| Map rendering | MapLibre GL with heatmap/cluster layers | Already in place, supports all needed visualizations |

---

## 13. Out of Scope (for this iteration)

- Real AVL/GPS hardware integration (use simulated bus stream)
- Real AFC/fare transaction feed (use synthetic data)
- Real APC/occupancy sensors (use simulated occupancy)
- GTFS schedule import (build from OSM data)
- Real dispatch system integration (manual workflow via UI)
- Mobile app (web-only for now)
- Multi-tenant/multi-city support