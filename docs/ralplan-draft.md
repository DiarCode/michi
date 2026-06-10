# RALPLAN v2: Michi Platform Professional Hardening & Feature Enhancement

## Principles

1. **Correctness first** — Silent wrong predictions are worse than security gaps in a thesis context
2. **Minimal diff per step** — Each item is independently reviewable and deployable
3. **Test what you ship** — Every fix needs test coverage
4. **Don't reimplement what exists** — Verify before building
5. **Preserve existing behavior** — Refactor, don't rewrite

## Decision Drivers

1. **Prediction correctness** — Unnormalized features produce silently wrong outputs
2. **Resource safety** — Session leaks crash workers under load
3. **Developer velocity** — Monolithic `main.py` slows every change
4. **User trust** — No tests, no error handling, silent failures

---

## Phase 1a: P0a — Correctness (Highest Priority)

### 1a.1 Feature Normalizer Fix (CRITICAL)
**Why #1**: Every prediction served to the dashboard uses unnormalized features on a model trained with z-score normalization. All predictions are silently wrong.

**What**:
- Port `FeatureNormalizer` from `main.py:219-248` to `backend/ml/normalizer.py`
- Modify `load_model()` in `predictor.py` to extract and restore normalizer state from checkpoint artifact
- Cache normalizer alongside model in `get_cached_model()`
- Apply `norm.transform()` to features in `generate_predictions()` BEFORE feeding to model
- Fix `predictor.py:47` session leak: remove outer `SessionLocal()` at line 47, reuse inner session pattern

**Files**: `backend/ml/normalizer.py` (new), `backend/ml/predictor.py`, `backend/ml/data_loader.py`
**Acceptance**: Unit test feeds synthetic data through `generate_predictions()` with and without normalization, confirms outputs differ. Integration test: `/api/v1/analytics/predictions` returns non-trivial predictions.
**Pre-mortem**: If normalizer fix changes prediction magnitudes, dashboard charts may render differently. Verify by comparing before/after outputs on a held-out test set.
**Complexity**: M

### 1a.2 DB Session Leak Audit & Fix
**What**: Fix ALL session leaks, not just routers. Five sites identified:
- `predictor.py:47` — `SessionLocal()` never closed (startup path, CRITICAL)
- `forecast_service.py:42-92` — `get_kpi_metrics` bare except swallows errors and may leak
- `predictor.py:60-61` — redundant second session creation when one was passed
- `tasks.py:30` — `_get_db_session()` helper, depends on caller
- `seed.py:66` — verify session cleanup

**Fix**: For each site, add try/finally with explicit `session.close()`. In `predictor.py`, remove the outer `db_session = SessionLocal()` at line 47 and always use the inner session pattern with proper cleanup.

**Files**: `backend/ml/predictor.py`, `backend/services/forecast_service.py`, `backend/tasks.py`, `backend/seed.py`
**Acceptance**: `pytest` with session leak detection (open 100 connections, verify all closed). No `SessionLocal()` call without matching `close()`.
**Pre-mortem**: If session cleanup is too aggressive, long-running queries may fail. Verify with concurrent load test.
**Complexity**: S

### 1a.3 Train Return Signature Fix
**What**: `train_offline` returns `(model, metrics, norm)` but some Streamlit callers may expect `(model, metrics)`. Audit all call sites and fix.

**Files**: `main.py` (Streamlit UI section)
**Acceptance**: Streamlit training flow works end-to-end without unpacking errors.
**Complexity**: S

---

## Phase 1b: P0b — Security & Reliability

### 1b.1 CORS Hardening
**What**: Replace `allow_origins=["*"]` with env-configurable whitelist.

**Implementation**:
- Add `ALLOWED_ORIGINS` env var (comma-separated, default `http://localhost:3100,http://localhost:5173,http://localhost:8600`)
- Parse in `app.py` lifespan, pass to `CORSMiddleware`
- Add `.env.example` with `ALLOWED_ORIGINS` documented

**Files**: `backend/app.py`
**Acceptance**: `curl -I -H "Origin: http://evil.com" -H "Access-Control-Request-Method: GET" -X OPTIONS http://localhost:8100/api/v1/stations` does NOT include `Access-Control-Allow-Origin: http://evil.com`. `curl -I -H "Origin: http://localhost:3100" ...` DOES include `Access-Control-Allow-Origin: http://localhost:3100`.
**Complexity**: S

### 1b.2 WebSocket Auth (Shared-Secret)
**What**: Add token-based auth to `/ws/realtime` using shared-secret comparison.

**Implementation**:
- Accept `token` query param on WS connect
- Validate `token == WS_AUTH_SECRET` env var
- Reject with 4001 close code if mismatch
- Frontend sends `?token=<secret>` when connecting
- If `WS_AUTH_SECRET` env var is not set, allow all connections (dev mode)

**Files**: `backend/websocket.py`, `dashboard/src/lib/websocket.ts`, `dashboard/src/hooks/useWebSocket.ts`
**Acceptance**: Unauthenticated WS connection is rejected with 4001 when `WS_AUTH_SECRET` is set. Authenticated connection receives data. No auth required when env var unset.
**Complexity**: S

### 1b.3 API Error Handling
**What**: Add structured exception handling. Framing: routers have **inconsistent** error handling (some have 404s, most have none).

**Implementation**:
- Create `backend/exceptions.py` with `AppException` base class and `NotFoundException`, `ValidationException`
- Add global exception handler in `app.py` returning consistent JSON: `{"detail": "...", "status": 404}`
- Add request logging middleware
- Fix routers that lack error handling (analytics, dashboard, depot, executive, passenger)

**Files**: `backend/exceptions.py` (new), `backend/app.py`, all `backend/routers/*.py`
**Acceptance**: `GET /api/v1/stations/nonexistent` returns `{"detail": "Station nonexistent not found", "status": 404}`. All 5xx errors are logged.
**Complexity**: M

### 1b.4 File Upload Security
**What**: Add size limit, type validation, and schema validation to `/analytics/upload`.

**Implementation**:
- Add `max_length=10_000_000` (10MB) to `UploadFile`
- Validate `content_type` is CSV
- Validate CSV headers match expected schema
- Cap rows at 10,000
- Actually persist data to `HistoricalRidershipORM` or return clear error

**Files**: `backend/routers/analytics.py`
**Acceptance**: Upload of >10MB file returns 413. Upload of non-CSV returns 415. Upload of valid CSV with wrong headers returns 422.
**Complexity**: S

### 1b.5 Connection State Exposure (NOT reconnect — that already exists)
**What**: Expose WSClient connection state and add UI indicator. WS reconnect already works in `websocket.ts:78-91` with exponential backoff. What's missing: connection state visibility.

**Implementation**:
- Add `state: "connecting" | "connected" | "disconnected"` public getter to `WSClient`
- Emit state change callbacks (`onstatechange`)
- In `useWebSocket.ts`, sync state to `connectionStore`
- Add `ConnectionIndicator` component in dashboard header (green dot = connected, yellow = connecting, red = disconnected)

**Files**: `dashboard/src/lib/websocket.ts`, `dashboard/src/hooks/useWebSocket.ts`, `dashboard/src/stores/connectionStore.ts`, new `dashboard/src/components/ConnectionIndicator.tsx`, `dashboard/src/App.tsx`
**Acceptance**: Kill backend → indicator turns red → restart backend → indicator turns green within 30s. No reimplementation of existing reconnect logic.
**Complexity**: S

### 1b.6 API Error Interceptor (Frontend)
**What**: Add axios response interceptor and connection status awareness.

**Implementation**:
- Add axios response interceptor for 5xx errors (show toast)
- Add retry logic for transient failures (3 retries, exponential backoff)
- Connection status reads from `connectionStore`

**Files**: `dashboard/src/lib/api.ts`
**Acceptance**: Backend down → frontend shows "Connection lost" toast. Backend recovers → requests succeed after retry.
**Complexity**: S

---

## Phase 2: P1 — Quality & UX

### 2.1 ML Pipeline Unit Tests
- `FeatureNormalizer`: fit/transform/inverse_transform roundtrip, zero-std handling
- `PageHinkley`: drift detection, reset
- `ResidualKalman`: predict/update cycle, dimension checks
- `reconcile_mint`: coherence improvement after reconciliation
- `nb_nll`: non-negative loss, gradient exists
- `build_astana_network`: valid NetworkSpec
- `build_hierarchy`: S matrix sums correctly
**Complexity**: M

### 2.2 Frontend Test Infrastructure
- Install vitest + @testing-library/react
- Test `api.ts` (mock axios, verify calls)
- Test `simulationStore.ts` (state transitions)
- Test `useWebSocket` hook (connect, disconnect, state)
**Complexity**: M

### 2.3 Remove Dead Code
- Remove `load_dataset_csv` (always returns None)
- Audit `RichLogger` for unused methods
- Remove unused imports in `main.py`
**Complexity**: S

### 2.4 Add Linting
- Add ruff config to `pyproject.toml`
- Add mypy config
- Add ESLint for dashboard
- Run `ruff check` and fix violations
**Complexity**: S

### 2.5 Station Detail Panel
- Click station on map → panel shows hourly load, forecast, alerts, connected routes
- Data from `/stations/{id}/detail` API
**Complexity**: M

### 2.6 Route Filter on Map
- Route dropdown above map
- Highlights stops and path on select
**Complexity**: M

---

## Phase 3: P2 — Observability & Polish

### 3.1 Structured Logging
- Replace `print()` with `logging` module across backend
- JSON formatter, request ID middleware
**Complexity**: M

### 3.2 Health Check Enrichment
- DB connectivity, model loaded, Redis ping, uptime
**Complexity**: S

### 3.3 Alert Rule Engine
- Threshold rules: crowding, low demand, model drift
- Celery beat task every 5 minutes
**Complexity**: M

### 3.4 Heatmap Overlay on Map
- MapLibre GL JS heatmap layer
- Data from station ridership
**Complexity**: M

### 3.5 Time-of-Day Slider
- Hour slider (0-23) on map
- Updates station markers and forecasts
**Complexity**: M

### 3.6 Dark Mode
- Zustand store + localStorage persistence
- CSS variables, theme toggle in header
**Complexity**: S