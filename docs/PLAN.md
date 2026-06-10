# Michi — Improvement Plan

## Completed ✅

### P0a: Correctness (Completed 2026-06-09)
- [x] **1a.1: Feature Normalizer Fix** — Ported `FeatureNormalizer` to `backend/ml/normalizer.py`. Modified `load_model()` to load normalizer from checkpoint. Applied `norm.transform()` in `generate_predictions()`. Fixed `predictor.py:47` session leak.
- [x] **1a.2: DB Session Leaks** — Audited all `SessionLocal()` calls. Fixed `predictor.py` (own_session pattern with finally/close). Improved `forecast_service.py` error logging. All routers use `Depends(get_db_session)`.
- [x] **1a.3: Train Return Signature** — Audited all call sites. Both already unpack `(model, metrics, norm)` correctly.

### P0b: Security & Reliability (Completed 2026-06-09)
- [x] **1b.1: CORS Hardening** — Replaced `allow_origins=["*"]` with `ALLOWED_ORIGINS` env var. Added `.env.example`. Verified with origin parsing tests.
- [x] **1b.2: WebSocket Auth** — Added `WS_AUTH_SECRET` env var with constant-time token comparison. Dev mode allows all when unset. Frontend sends `?token=` via `VITE_WS_TOKEN`.
- [x] **1b.3: Structured Error Handling** — Created `backend/exceptions.py` with `AppException`, `NotFoundException`, `ValidationException`, `PayloadTooLargeException`. Added global exception handler in `app.py`. Updated `stations.py` to use `NotFoundException`.
- [x] **1b.4: File Upload Security** — Added 10MB size limit, CSV content type validation, header validation, row count cap (10k), and graceful handling of malformed rows.
- [x] **1b.5: Connection State Exposure** — Added `state` property and `onStateChange` callbacks to `WSClient`. Created `ConnectionIndicator` component (green/yellow/red dot). Wired into App header.
- [x] **1b.6: API Error Interceptor** — Added axios response interceptor for 5xx errors with structured logging. Added retry interceptor for GET requests (3 retries, exponential backoff).

---

## Next (P1)

### P1: Test Coverage
- [ ] **T1: ML Pipeline Tests** — Unit tests for FeatureNormalizer, PageHinkley, ResidualKalman, reconcile_mint, nb_nll, build_astana_network
- [ ] **T2: Frontend Test Infra** — Vitest + React Testing Library. Test api.ts, stores, useWebSocket hook.
- [ ] **T3: Integration Test** — Seed → train → predict → API → dashboard smoke test

### P1: Code Quality
- [ ] **Q1: Type Safety** — Add type hints to `main.py` boundaries
- [ ] **Q2: Dead Code Removal** — Remove `load_dataset_csv`, audit RichLogger
- [ ] **Q3: Linting** — Add ruff + mypy config

### P1: Frontend UX
- [ ] **F3: Station Detail Panel** — Click station → hourly load, forecast, alerts, routes
- [ ] **F4: Route Filter on Map** — Dropdown highlights route stops and path

---

## Later (P2)

### P2: Observability & Ops
- [ ] **O1: Structured Logging** — Replace `print()` with `logging` module
- [ ] **O2: Health Check Enrichment** — DB, model, Redis status
- [ ] **O3: Metrics Endpoint** — Prometheus format

### P2: Data & ML
- [ ] **D1: TimescaleDB Migration** — SQLite → TimescaleDB
- [ ] **D2: Model Versioning API** — `/api/v1/models` endpoints
- [ ] **D3: Alert Rule Engine** — Threshold-based auto-generation

### P2: Frontend Polish
- [ ] **F5: Heatmap Overlay** — Station load heatmap on map
- [ ] **F6: Time-of-Day Slider** — Hour filter for map data
- [ ] **F7: Dark Mode** — CSS variables + Zustand persist

---

## Known Issues for Follow-up

- **Feature dimension mismatch**: Checkpoint normalizer has 14 features but `data_loader.py` produces 11. Normalizer is correctly skipped with warning when dimensions don't match. Data loader needs to be updated to produce all 14 features.
- **DB migration needed**: `model_artifacts` table doesn't exist in SQLite. Needs Alembic migration.
- **`main.py` decomposition**: 3277-line monolith still needs to be split (P1 task).

## Changelog

- **2026-06-09**: Completed all P0 tasks (1a.1–1a.3, 1b.1–1b.6). Created `docs/CLAUDE.md` architecture doc and `docs/PLAN.md` improvement plan.