# Michi Platform Comprehensive Enhancement Plan (v2)

**Created:** 2026-05-31 | **Revised:** 2026-05-31 (Iteration 2)
**Scope:** Real-time simulation stream, timeline bar, full bug fixes, frontend/backend enhancements
**Estimated Complexity:** HIGH (5 phases, ~55 files touched)

---

## RALPLAN-DR Summary

### Principles

1. **Stabilize before extend** — Fix all backend bugs and data integrity issues before building new features. Block Phase 2 on Phase 1 completion with verified acceptance criteria.
2. **Single WebSocket channel with subscription filtering** — All real-time communication on one `/ws/realtime` connection using typed event envelopes. Clients declare which event types they want on connect. This avoids bandwidth waste without multi-socket complexity.
3. **Pull for timeline, push for live** — Timeline scrubbing is user-driven (REST pull); live bus/simulation positions are continuous (WebSocket push). When user scrubs timeline, live WS updates pause; releasing scrubber resumes live with latest tick.
4. **Adopt what is installed** — Zustand for WebSocket/UI state + TanStack Query for REST server state (both already in package.json). Celery for CPU-bound simulation (already installed with Redis broker). Recharts for charts (already installed). Do NOT introduce new libraries.
5. **DB as source of truth** — Replace all in-memory data stores with database-backed queries. Eliminate dual-store patterns. Simulation state checkpointed to DB every 60 ticks for crash recovery.

### Decision Drivers (top 3)

1. **Data integrity** — Route ID mismatches ("R12" vs "Route_12"), mock-vs-real inconsistencies, and in-memory/DB divergence cause cascading failures. Must resolve with canonical format: **"R12"** (matches DB foreign keys).
2. **Real-time simulation fidelity** — The P0 feature requires a simulation engine that produces plausible passenger flow data, streams it via WebSocket, and supports time-arbitrary queries. Celery handles CPU-bound model inference; lightweight async relay bridges to WebSocket clients.
3. **Frontend observability** — Silent error swallowing, missing error boundaries, no loading states. Zustand/TanStack Query boundary must be explicit: Zustand = WebSocket-derived + UI ephemeral; TanStack Query = REST server state with cache/refetch semantics.

### Viable Options

#### Option A: Integrated WebSocket + REST Timeline (CHOSEN)

Extend existing `/ws/realtime` with subscription filtering + new event types. Simulation engine runs as Celery task (CPU-bound DTS-GSSF inference), publishes to Redis, lightweight async listener relays to WebSocket. REST `/api/v1/timeline` for scrubber queries.

| Aspect | Assessment |
|--------|-----------|
| Pros | Single WS connection; Celery avoids event-loop blocking; subscription filtering prevents bandwidth waste; REST timeline supports caching/pagination |
| Cons | Redis relay adds ~50ms latency per tick (acceptable for 1Hz simulation); timeline REST calls add latency on scrub (mitigated by range prefetching) |

#### Option B: Dual WebSocket Channels (CONSIDERED AND REJECTED)

Separate `/ws/simulation` alongside `/ws/realtime`.

| Aspect | Assessment |
|--------|-----------|
| Rejection rationale | Two connections per client; double lifecycle management; Nginx must proxy both; no clear benefit since subscription filtering on a single channel achieves the same efficiency. Adds infrastructure complexity without proportional gain. |

#### Option C: SSE for Simulation + WebSocket for Positions (CONSIDERED AND REJECTED)

| Aspect | Assessment |
|--------|-----------|
| Rejection rationale | SSE is unidirectional (simulation needs control messages: start/stop/pause); third real-time protocol in stack; Nginx SSE proxying requires specific buffering config. Does not solve a problem Option A cannot handle. |

---

## Resolved Open Questions

**Q1: Canonical route ID format?** → **"R12"** (matches DB foreign keys, ORM models, seed.py). All WebSocket payloads, frontend code, and mock data must use this format. The WebSocket's `"Route_12"` format is the outlier and must be updated.

**Q2: Alert migration strategy?** → Alerts are **ephemeral** until DB-backed. No migration of in-memory state. On restart, alerts regenerate from threshold rules via `POST /api/v1/alerts/generate`. The in-memory `ALERTS` list and `_acked` set are replaced entirely with DB queries.

**Q3: Timeline-tick divergence UX contract?** → When user grabs timeline scrubber, live WS updates to the map **pause** (last-received tick cached). Map shows timeline-positioned data from REST API. When user releases scrubber, UI **snaps to latest tick** and resumes live WS updates. Visual indicator ("LIVE" badge vs "HISTORICAL" badge) shows current mode.

**Q4: Canonical WebSocket connection pattern?** → `useWebSocket` hook is canonical. LiveMap's direct `wsClient.connect()` call is removed. The hook provides cleanup, automatic connection management, and event routing to Zustand stores.

**Q5: Simulation tick rate and latency budget?** → Tick interval: **1 second** simulation time. Max acceptable delivery latency: **500ms** from tick generation to WebSocket receipt. If latency exceeds 500ms, UI shows "STALE" indicator.

**Q6: Simulation crash recovery?** → Simulation state (tick counter, Kalman filter state, drift detector state) is checkpointed to `model_artifacts` DB table every **60 ticks** (~1 minute). On Celery task restart, it reads the last checkpoint and resumes. If no checkpoint exists, simulation starts fresh.

---

## ADR: Integrated WebSocket with Subscription Filtering + Celery Simulation

- **Decision:** Extend `/ws/realtime` with subscription filtering; Celery for simulation; REST `/api/v1/timeline` for scrubber.
- **Drivers:** Data integrity, real-time fidelity, frontend observability.
- **Alternatives considered:** Dual WebSocket (B), SSE+WebSocket (C).
- **Why chosen:** Single connection with filtering is simpler than dual sockets; Celery avoids event-loop blocking; REST is correct for user-driven scrubbing.
- **Consequences:** WebSocket must support subscription handshake; Celery+Redis adds ~50ms relay latency; timeline endpoint must support efficient range queries.
- **Follow-ups:** Add WebSocket schema validation; add timeline data caching with 5-minute TTL; add reconnection with exponential backoff and state catch-up.

---

## Phase Dependency Graph

```
Phase 1 (Backend Fixes) [BLOCKING]
   |
   v
Phase 2 (Simulation Service + Timeline API) ──> Phase 3 (Frontend Fixes + State)
   |                                                    |
   v                                                    v
Phase 4 (Timeline + LiveMap + Simulation Dashboard) <---
   |
   v
Phase 5 (Polish + Integration Verification)
```

---

## Phase 1: Backend Foundation Fixes [BLOCKING — must pass all acceptance criteria before Phase 2 begins]

**Goal:** Eliminate all backend bugs, data integrity issues, and infrastructure gaps.

**Depends on:** Nothing (start immediately)

### Step 1.1: Fix infrastructure and database gaps

| File | Change |
|------|--------|
| `backend/requirements.txt` | Create with all imports from backend/ (fastapi, uvicorn, sqlalchemy, celery[redis], redis, websockets, numpy, torch, scikit-learn, pydantic, etc.) |
| `alembic/versions/002_add_missing_tables.py` | Migration for `historical_ridership`, `weather_readings`, `events`, `interventions`, `model_artifacts`, `prediction_accuracy` tables |
| `alembic/versions/003_alert_rich_fields.py` | Migration for alert rich fields (`family`, `what`, `when_hint`, `where_hint`, `why`, `confidence`, `consequence_if_ignored`, `sla_timer_minutes`, `acknowledged`, `assigned_to`) and forecast fields (`horizon_minutes`, `route_id`) |
| `backend/database.py` | Remove `Base.metadata.create_all()` call from `init_db()` after Alembic migrations are in place. Add `get_db_session()` generator for FastAPI `Depends()` injection. |

**Acceptance criteria:**
- AC1.1: `pip install -r backend/requirements.txt` succeeds without errors
- AC1.2: `alembic upgrade head` creates all 12 tables; `alembic downgrade base` drops all 12
- AC1.3: No `Base.metadata.create_all()` remains in `database.py`; all table creation is via Alembic

### Step 1.2: Fix data integrity and code bugs

| File | Change |
|------|--------|
| `backend/services/alert_service.py` | Replace in-memory `ALERTS` list with DB-backed queries via SQLAlchemy. Remove `_acked` set; use `AlertORM.acknowledged` column. Remove `_next_id`; use DB auto-increment. Accept ephemeral loss on migration (alerts regenerate from rules). |
| `backend/services/suggestion_service.py` | Fix line 106: `predicted` → `pred["predicted"]` in low-demand loop |
| `backend/ml/data_loader.py` | Remove dead code where transpose is immediately overwritten (lines 114-116) |
| `backend/websocket.py` | Replace `MOCK_BUSES` (2 buses, "Route_12"/"Route_34" format) with `realtime_service.BUS_POOL` (8 buses, "R12" format). Import `get_current_positions()` from `realtime_service.py`. Fix route ID format to "R12" canonical. |
| `backend/services/realtime_service.py` | Ensure all bus route_ids use "R12" format (not "Route_12"). Add `BUS_POOL` as exported constant. |
| `backend/routers/dashboard.py` | Replace hardcoded `on_time_performance: 94.2` with computed value from DB or configurable env var |
| `backend/routers/analytics.py` | Replace random data generation in trends/compare with DB queries on `HistoricalRidershipORM` and `PredictionAccuracyORM`. If no data, return empty arrays with `note` field. No `random` module usage. |
| `backend/routers/routes.py` | Remove `MOCK_ROUTES` (R1-R5) and `ROUTE_STOPS` fallback. All route data from DB (seeded as R12, R18, R25, R31, R40). |

**Acceptance criteria:**
- AC1.4: Alert CRUD operations read/write from DB; no in-memory state variables (`ALERTS`, `_acked`, `_next_id`) exist
- AC1.5: All route IDs across backend use "R12" format; grep for "Route_12" and "R1"/"R2"/"R3"/"R4"/"R5" returns zero results
- AC1.6: `analytics/trends` and `analytics/compare` return same results on repeated calls (no randomness)
- AC1.7: WebSocket broadcasts 8 buses (not 2) with correct "R12" route IDs

### Step 1.3: Fix session management and WebSocket infrastructure

| File | Change |
|------|--------|
| `backend/database.py` | Add `get_db_session()` generator yielding a session, closing it, suitable for FastAPI `Depends()` |
| `backend/routers/*.py` | Replace all ad-hoc `SessionLocal()` calls with `db: Session = Depends(get_db_session)` |
| `backend/services/alert_service.py`, `accuracy_service.py`, `artifact_store.py`, `intervention_service.py` | Replace internal `SessionLocal()` sessions with explicit session parameter passed from router layer |
| `backend/websocket.py` | Add subscription filtering to `ConnectionManager`: clients send `{"subscribe": ["bus_position", "simulation_tick", ...]}` on connect. `broadcast()` respects per-connection subscriptions. Default: all event types. |
| `dashboard/src/lib/websocket.ts` | Add exponential backoff reconnection (base 3s, max 30s, jitter). Add state catch-up: on reconnect, client sends `{"last_tick": N}` and server responds with missed `simulation_tick` events (up to 60 most recent from Redis buffer). |

**Acceptance criteria:**
- AC1.8: No bare `SessionLocal()` calls in router code; all sessions use `Depends(get_db_session)`
- AC1.9: `ConnectionManager` tracks per-connection subscriptions; `broadcast()` only sends to connections subscribed to that event type
- AC1.10: Frontend WebSocket reconnects with exponential backoff; on reconnect, receives up to 60 missed ticks

---

## Phase 2: Real-Time Simulation Service + Timeline API

**Goal:** When the app starts, begin streaming simulated passenger flow data for all stations with live model validation metrics.

**Depends on:** Phase 1 (all acceptance criteria pass)

### Step 2.1: Simulation engine (Celery task)

| File | Change |
|------|--------|
| `backend/services/simulation_service.py` (NEW) | `SimulationEngine` class: (1) generates realistic hourly ridership per station using sinusoidal rush-hour patterns + noise + weather/holiday effects from DB, (2) runs DTS-GSSF model inference on generated data (via `predictor.py`), (3) computes real-time validation metrics (MAE, MAPE, accuracy) comparing predictions vs simulated actuals, (4) detects drift when MAPE exceeds 15% threshold, (5) checkpoints state to DB every 60 ticks. Each tick: timestamp, station ridership snapshot, forecast values, confidence intervals, drift status. |
| `backend/tasks.py` | Add `run_simulation` Celery task that instantiates `SimulationEngine` and runs tick loop. Publishes each tick to Redis channel `michi:simulation`. Checkpoints state to `model_artifacts` table every 60 ticks. On restart, reads last checkpoint. |
| `backend/websocket.py` | Add async `simulation_relay()` listener in lifespan that subscribes to Redis `michi:simulation` channel and broadcasts `simulation_tick`, `validation_metric`, `drift_alert` events to WebSocket clients (respecting subscriptions). Replace `mock_bus_stream()` with combined `combined_stream()` that handles both bus positions and simulation relay. |

**Acceptance criteria:**
- AC2.1: `celery -A backend.tasks worker` starts without errors; `run_simulation` task produces 1 tick/second
- AC2.2: Each tick contains data for all 374 stations (not a subset)
- AC2.3: Validation metrics converge: MAPE < 15% within 100 ticks for pattern-based simulation
- AC2.4: Simulation state checkpointed to DB; after `run_simulation` restart, resumes from last checkpoint
- AC2.5: WebSocket clients subscribed to `simulation_tick` receive events within 500ms of tick generation

### Step 2.2: Simulation API endpoints

| File | Change |
|------|--------|
| `backend/routers/simulation.py` (NEW) | `POST /api/v1/simulation/start` (trigger Celery task), `POST /api/v1/simulation/stop` (revoke task), `GET /api/v1/simulation/state` (current metrics + tick + running status), `GET /api/v1/simulation/metrics` (historical MAE/MAPE time series from checkpoints) |
| `backend/app.py` | Register `simulation` router. In `lifespan`, auto-start simulation via `run_simulation.delay()` alongside WebSocket. |

**Acceptance criteria:**
- AC2.6: `POST /simulation/start` returns 202 with task_id; `GET /simulation/state` shows `running: true`
- AC2.7: `POST /simulation/stop` returns 200; `GET /simulation/state` shows `running: false`
- AC2.8: `GET /simulation/metrics` returns time series of MAE/MAPE from checkpoints

### Step 2.3: Timeline data API

| File | Change |
|------|--------|
| `backend/routers/timeline.py` (NEW) | `GET /api/v1/timeline` with query params: `station_id` (optional), `start_time`, `end_time`, `resolution` (5min/15min/1h). Returns `[{timestamp, actual, predicted, confidence_upper, confidence_lower}]`. Past: `actual` from `HistoricalRidershipORM`/`RidershipORM`. Future: `actual=null`, `predicted` from `ForecastORM` or simulation. |
| `backend/app.py` | Register `timeline` router. |

**Acceptance criteria:**
- AC2.9: Timeline endpoint returns continuous series; `actual` non-null for past, null for future
- AC2.10: Query with `resolution=15m` and 24h range returns 96 data points per station
- AC2.11: Response time < 200ms for single-station 24h query; < 2s for all-stations query

---

## Phase 3: Frontend Foundation Fixes + State Management

**Goal:** Fix all frontend bugs, adopt Zustand for WebSocket/UI state (keeping TanStack Query for REST), add error handling infrastructure.

**Depends on:** Phase 1 (backend API consistency)

### Step 3.1: Fix role system and route mismatches

| File | Change |
|------|--------|
| `dashboard/src/components/layout/RoleGuard.tsx` | Use `michi-role` localStorage key (not `michi_role`). Use `UserRole` type from `types/index.ts` (dispatch/research/planning/executive/depot/passenger). |
| `dashboard/src/lib/constants.ts` | Add `SEED_ROUTE_IDS = ["R12", "R18", "R25", "R31", "R40"]`. Remove any R1-R5 references. |
| `dashboard/src/routes/CommandCenter.tsx` | Remove `/route-command` alias or make it redirect to `/`. |

**Acceptance criteria:**
- AC3.1: RoleGuard reads `michi-role`; role values match `UserRole` union type; no "Dispatch Manager"/"City Planner" strings
- AC3.2: No "R1"/"R2"/"R3"/"R4"/"R5" route ID strings exist in frontend code
- AC3.3: `/route-command` redirects to `/` or shows distinct content

### Step 3.2: Zustand stores (WebSocket/UI state only) + TanStack Query (REST state)

**Boundary definition:**
- **Zustand owns:** Bus positions (from WS), simulation tick state, timeline scrubber position, selected station/route UI state, WebSocket connection status
- **TanStack Query owns:** Station list, route list, forecasts, KPIs, alerts, analytics, interventions, executive data, depot data, passenger data — all REST-fetched data with cache/refetch semantics

| File | Change |
|------|--------|
| `dashboard/src/stores/busStore.ts` (NEW) | `buses: Record<string, BusPosition>`, `subscribeToBusPositions()`. Updates from `bus_position` WS events. |
| `dashboard/src/stores/simulationStore.ts` (NEW) | `isRunning`, `currentTick`, `metrics` (MAE/MAPE/accuracy time series), `driftStatus`, `isStale`. Actions: `startSimulation()`, `stopSimulation()`, `updateFromTick()`, `markStale()`. |
| `dashboard/src/stores/timelineStore.ts` (NEW) | `currentTime`, `isPlaying`, `playSpeed`, `range` (start/end), `data: TimelinePoint[]`, `mode: "live" | "historical"`. Actions: `scrubTo(timestamp)`, `play()`, `pause()`, `setSpeed()`, `enterLiveMode()`, `enterHistoricalMode()`. |
| `dashboard/src/stores/connectionStore.ts` (NEW) | WebSocket connection status: `connected`, `lastTickReceived`, `reconnectAttempt`. Used for stale detection. |

**Acceptance criteria:**
- AC3.4: All WebSocket-derived state lives in Zustand stores; no `useState` for bus/simulation/timeline data
- AC3.5: All REST API calls remain in TanStack Query hooks; no REST fetching logic in Zustand stores
- AC3.6: When WS events update server state (e.g., new alert), the store invalidates the relevant TanStack Query key

### Step 3.3: Error handling, loading states, and cleanup

| File | Change |
|------|--------|
| `dashboard/src/components/ui/toast.tsx` (NEW) | shadcn/ui toast with variants: success, error, warning, info. Auto-dismiss 5s. |
| `dashboard/src/lib/toast.ts` (NEW) | `showToast(message, variant)` imperative API. |
| `dashboard/src/components/ErrorBoundary.tsx` (NEW) | React error boundary: catches render errors, shows fallback with retry button, logs error. |
| All `dashboard/src/routes/*.tsx` | Replace `.catch(() => {})` with `.catch((err) => showToast(err.message, "error"))`. Add loading skeletons. |
| `dashboard/src/routes/LiveMap.tsx` | Remove direct `wsClient.connect()` call (line 36-43). Use `useWebSocket` hook exclusively. |
| `dashboard/package.json` | Move `@playwright/test` to devDependencies. Keep `zustand` and `recharts` (now used). Remove `RoutePolyline.tsx` if not integrated. |

**Acceptance criteria:**
- AC3.7: Grep for `catch(() => {})` in `dashboard/src/` returns zero results
- AC3.8: Every route page shows a loading skeleton while TanStack Query data is `isLoading`
- AC3.9: ErrorBoundary wraps the app root; crashed components show fallback UI, not white screen
- AC3.10: No direct `wsClient.connect()` calls in route components (only `useWebSocket` hook)

### Step 3.4: Wire up unreachable routes

| File | Change |
|------|--------|
| `dashboard/src/App.tsx` ROLE_NAV | Add `/forecast` (research, planning), `/timetable` (passenger), `/settings` (all roles) to nav. |
| `dashboard/src/components/dashboard/ForecastChart.tsx` | Integrate into ForecastPage using Recharts `LineChart`. |
| `dashboard/src/components/map/RoutePolyline.tsx` | Integrate into MapContainer as MapLibre `LineLayer` (not raw SVG). If too complex for this phase, move to Phase 4. |

**Acceptance criteria:**
- AC3.11: `/forecast`, `/timetable`, `/settings` appear in sidebar for at least one role
- AC3.12: ForecastChart renders in ForecastPage with Recharts

---

## Phase 4: Timeline Bar + LiveMap Enhancement + Simulation Dashboard

**Goal:** Build the timeline scrubber on LiveMap and the real-time model validation display.

**Depends on:** Phase 2 (simulation service + timeline API) + Phase 3 (Zustand stores + error handling)

### Step 4.1: Timeline Bar component

| File | Change |
|------|--------|
| `dashboard/src/components/map/TimelineBar.tsx` (NEW) | Horizontal timeline scrubber: (1) past segment (solid grey), current marker (blue vertical line + time label), future segment (dashed purple); (2) draggable scrubber handle; (3) time display (date + HH:MM) at current position; (4) play/pause button; (5) speed selector (1x, 2x, 5x); (6) confidence interval band for future (semi-transparent purple fill); (7) "LIVE" badge when in live mode, "HISTORICAL" badge when scrubbing. Reads from `timelineStore`. Fetches from `/api/v1/timeline`. |
| `dashboard/src/hooks/useTimeline.ts` (NEW) | Hook: fetches timeline data from API, manages range prefetching (fetch ±2h around current position), syncs with `timelineStore`. When user grabs scrubber, calls `timelineStore.enterHistoricalMode()`. When released, calls `timelineStore.enterLiveMode()`. |

**Acceptance criteria:**
- AC4.1: Timeline bar renders below LiveMap map area; shows past (grey), current (blue marker), future (dashed purple)
- AC4.2: Dragging scrubber to past shows historical actual ridership in station popups
- AC4.3: Dragging scrubber to future shows predicted ridership + confidence interval in station popups
- AC4.4: Auto-play advances timeline at selected speed; "LIVE" badge visible
- AC4.5: Confidence band (upper/lower) rendered as semi-transparent area on future segment

### Step 4.2: Enhance LiveMap with timeline integration

| File | Change |
|------|--------|
| `dashboard/src/routes/LiveMap.tsx` | Add `TimelineBar` below map. Replace local `useState` for buses with `busStore`. When `timelineStore.mode === "historical"`, station data comes from timeline API response; when `"live"`, from WS-driven store. |
| `dashboard/src/components/map/StationMarker.tsx` | Visual states: past-data (grey outline), current (blue solid), future-prediction (purple dashed). Confidence badge on future markers. |
| `dashboard/src/components/map/MapContainer.tsx` | Pass timeline time context to markers. Update cluster/heatmap layer for timeline-positioned data. |

**Acceptance criteria:**
- AC4.6: Station markers change appearance based on timeline mode (live=blue, historical=grey, future=purple)
- AC4.7: Station popups show data for the timeline-selected time (not just "current" time)
- AC4.8: Heatmap layer reflects timeline-positioned data

### Step 4.3: Simulation validation dashboard

| File | Change |
|------|--------|
| `dashboard/src/components/dashboard/SimulationMetrics.tsx` (NEW) | Real-time validation panel: (1) MAE/MAPE/Accuracy number displays with trend arrows, (2) Recharts `LineChart` for MAE/MAPE time series, (3) drift status indicator (green < 10%, yellow 10-15%, red > 15%), (4) station prediction count. Reads from `simulationStore`. |
| `dashboard/src/routes/SimulationPage.tsx` | Integrate SimulationMetrics. Add start/stop controls. Show simulation state. Add per-station ridership grid (current vs predicted). |
| `dashboard/src/routes/CommandCenter.tsx` | Mini simulation status card: current MAE/MAPE + drift indicator + link to SimulationPage. |

**Acceptance criteria:**
- AC4.9: SimulationMetrics updates in real-time on each `validation_metric` WS event
- AC4.10: MAE/MAPE time series chart shows last 5 minutes of data
- AC4.11: Start/stop controls trigger `POST /simulation/start` and `POST /simulation/stop`
- AC4.12: CommandCenter shows mini simulation status card

### Step 4.4: Connect all WebSocket events

| File | Change |
|------|--------|
| `dashboard/src/lib/websocket.ts` | Extend `WSEvent` type: add `simulation_tick`, `validation_metric`, `drift_alert`, `alert`, `forecast_update`. Add subscription support: client sends `{"subscribe": [...]}` after connect. |
| `dashboard/src/hooks/useWebSocket.ts` | Route events to Zustand stores: `bus_position` → `busStore`, `simulation_tick` → `simulationStore`, `validation_metric` → `simulationStore`, `alert` → invalidate TanStack Query alert key, `drift_alert` → `simulationStore`. |

**Acceptance criteria:**
- AC4.13: Each WS event type routes to correct Zustand store; no events dropped
- AC4.14: Client subscribes to event types relevant to current page (e.g., SimulationPage subscribes to simulation events, not bus positions)

---

## Phase 5: Polish + Integration Verification

**Goal:** Final fixes, visual polish, end-to-end verification.

**Depends on:** Phase 4

### Step 5.1: Visual polish

| File | Change |
|------|--------|
| `dashboard/src/components/ui/badge.tsx` | Add dark mode variants. |
| All components | Audit dark mode contrast; fix any issues. |

**Acceptance criteria:**
- AC5.1: Badge renders correctly in both light and dark themes
- AC5.2: No contrast issues on any page in dark mode

### Step 5.2: Documentation and CI

| File | Change |
|------|--------|
| `README.md` | Fix corrupted sections. Add: project overview, quickstart (`docker-compose up`), architecture diagram, env vars, API summary. |
| `.github/workflows/ci.yml` (NEW) | CI: lint (ruff + eslint), type check (mypy/tsc), test (pytest), docker build check. |

**Acceptance criteria:**
- AC5.3: README renders correctly; quickstart instructions work
- AC5.4: CI runs on push; lint/type/test pass on clean branch

### Step 5.3: Type sync + frontend types for new features

| File | Change |
|------|--------|
| `dashboard/src/types/index.ts` | Add: `SimulationTick`, `ValidationMetric`, `DriftAlert`, `TimelinePoint`, `SimulationState`. Align `Alert` with `AlertORM` (add rich fields). |

**Acceptance criteria:**
- AC5.5: Frontend types match backend API response shapes; no `as any` casts needed

### Step 5.4: End-to-end verification

| Task | Acceptance Criteria |
|------|---------------------|
| `docker-compose up --build` | All containers start; no build errors; health check passes |
| Open dashboard at `localhost:3100` | App loads; no console errors; all nav routes render |
| Verify simulation starts on boot | WS receives `simulation_tick` events; SimulationMetrics updates |
| Scrub timeline on LiveMap | Past shows actual data; future shows predictions with confidence; auto-play works; LIVE/HISTORICAL badges toggle |
| Verify alert persistence | Create alert via threshold; restart backend; alert still present in DB |
| Verify route ID consistency | All route IDs "R12" format across frontend/backend/WS |
| Verify no silent errors | Grep `catch(() => {})` returns zero results |
| Verify WebSocket reconnect | Kill backend briefly; frontend reconnects with backoff; no stale state after reconnect |

---

## Guardrails

### Must Have
- Simulation streams data for ALL stations, not a subset
- Timeline scrubber shows clear visual distinction between past/current/future
- All backend bugs fixed with testable acceptance criteria
- Docker build succeeds end-to-end
- No silent error swallowing in frontend
- "R12" canonical route ID format everywhere
- Celery for simulation (not asyncio.create_task)
- Zustand for WS/UI + TanStack Query for REST (clear boundary)

### Must NOT Have
- No new state management library (use Zustand, already installed)
- No new charting library (use Recharts, already installed)
- No separate WebSocket channel (extend existing `/ws/realtime`)
- No rewriting of the DTS-GSSF model code
- No authentication system (out of scope)
- No `asyncio.create_task` for CPU-bound simulation inference
- No TanStack Query replacement with Zustand for REST data

---

## Success Criteria

1. App starts and immediately streams simulation data for all 374 stations with live MAE/MAPE/accuracy metrics.
2. LiveMap timeline bar scrubs between past (actual data) and future (predicted data) with confidence intervals and LIVE/HISTORICAL mode badges.
3. All 12 backend bugs and 12 frontend bugs are resolved and verified against acceptance criteria.
4. `docker-compose up --build` succeeds with no errors.
5. No `catch(() => {})` patterns remain in the frontend.
6. All route IDs use "R12" format across frontend, backend, and WebSocket.
7. Simulation uses Celery (not asyncio); no event-loop blocking.
8. Zustand handles WS/UI state; TanStack Query handles REST state; boundary is explicit.