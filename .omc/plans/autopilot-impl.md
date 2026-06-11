# Michi Platform Implementation Plan

## Execution Order (Parallel Where Possible)

### Wave 1: Frontend Page Rewiring (5 parallel agents)
These are independent - each page can be wired in parallel:

- Agent A: Wire alerts.tsx → real API (P1-1)
- Agent B: Wire command-center.tsx → real API (P1-2)  
- Agent C: Wire executive.tsx → real API (P1-3)
- Agent D: Wire forecast.tsx → real API (P1-4)
- Agent E: Fix simulation store → REST API (P1-5)

### Wave 2: Backend Fixes (3 parallel agents)
Depends on Wave 1 completion for frontend contract awareness:

- Agent F: Connect Kalman + hierarchical to predictor (P1-7)
- Agent G: Persist alert rules to DB (P1-8)
- Agent H: Fix CSV upload persistence (P1-9)

### Wave 3: WebSocket & Integration (1 agent)
Must complete after Wave 1 (frontend) and Wave 2 (backend):

- Agent I: Wire WS alert/forecast_update invalidation (P1-6)

### Wave 4: Differentiator Features (2-3 parallel agents)
Independent features, can run after Wave 3:

- Agent J: Prediction confidence overlay (P2-1)
- Agent K: Auto-retrain pipeline (P2-2)
- Agent L: Interactive scenario engine (P2-4)

### Wave 5: B2C Expansion (2 parallel agents)
- Agent M: Passenger crowding page (P3-1)
- Agent N: Executive report + weather integration (P3-2 + P3-3)

## Key Architecture Decisions
1. All frontend pages use TanStack Query (React Query) for data fetching
2. Zustand stores only for WebSocket-driven real-time state
3. Backend changes maintain existing API contracts (no breaking changes)
4. New endpoints follow existing router pattern in backend/routers/
5. ML pipeline integration uses existing predictor.py get_cached_model() pattern
