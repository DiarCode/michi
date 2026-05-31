# Open Questions — RESOLVED

## michi-platform-enhancement - 2026-05-31

All questions resolved in plan v2 revision:

- [x] **Q1: Canonical route ID format?** → **"R12"** (matches DB foreign keys). All WS payloads, frontend code, and mock data must use this format. `"Route_12"` in websocket.py is the outlier.
- [x] **Q2: Alert migration strategy?** → Alerts are **ephemeral** until DB-backed. No migration of in-memory state. On restart, alerts regenerate from threshold rules via `POST /api/v1/alerts/generate`.
- [x] **Q3: Timeline-tick divergence UX contract?** → When user grabs scrubber, live WS updates **pause**; map shows timeline-positioned data from REST API. When released, UI **snaps to latest tick** and resumes live. "LIVE" / "HISTORICAL" badges indicate mode.
- [x] **Q4: Canonical WebSocket connection pattern?** → `useWebSocket` hook is canonical. LiveMap's direct `wsClient.connect()` is removed.
- [x] **Q5: Simulation tick rate and latency budget?** → 1 second simulation time. Max 500ms delivery latency. Exceeding shows "STALE" indicator.
- [x] **Q6: Simulation crash recovery?** → State checkpointed to `model_artifacts` DB table every 60 ticks (~1 min). On Celery restart, reads last checkpoint and resumes. Fresh start if no checkpoint.

## Implementation Notes (from Architect/Critic review)

- `redis.asyncio` must be used for `simulation_relay()` (async Redis client, available in `redis>=4.2`)
- `/route-command` will simply redirect to `/` (simplest option)
- Checkpoint gap of up to 60 ticks on crash is acceptable for thesis project scope