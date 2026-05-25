# Michi Transit Platform — Deep Domain Analysis & Improvement Plan

## 1. Current State Assessment

### What Works
- **Streamlit research UI**: Full DTS-GSSF pipeline (data gen → train → online sim → forecast → compare)
- **FastAPI backend**: 6 ORM models, 5 routers, WebSocket bus streaming, mock data
- **React dashboard**: 6 routes (CommandCenter, LiveMap, Alerts, Scenarios, Reports, Settings)
- **Data exports**: Parquet, GTFS, GeoJSON, CSV, TorchScript
- **28 passing tests**: ORM CRUD, services, API integration

### Critical Gaps (Transit Domain Perspective)

| # | Gap | Why It Matters | Priority |
|---|-----|---------------|----------|
| G1 | **No real-time passenger counting input** | Core value is predicting flow, but no way to input actual ridership to validate/correct | P0 |
| G2 | **Map has no route selection/filtering** | Operator can't click a line to see stops, schedule, load — must know station names | P0 |
| G3 | **No heatmap layer on map** | CongestionHeatmap component exists but isn't overlaid on the actual map | P0 |
| G4 | **No time-of-day filter** | Rush hour vs off-peak is the #1 driver of variation — no time slider exists | P0 |
| G5 | **Bus markers show position only** | No occupancy %, speed, ETA to next stop, or delay indicator | P1 |
| G6 | **No station detail panel** | Clicking a station should show: hourly load, forecast, alerts, connected routes | P1 |
| G7 | **Alerts are mock-only** | No rule engine to auto-generate alerts from thresholds | P1 |
| G8 | **Scenario planner is basic** | No before/after comparison, no ridership impact estimate | P1 |
| G9 | **No schedule/timetable view** | Transit operators live by timetables — no schedule display exists | P2 |
| G10 | **Reports page is stub** | No actual report generation or export | P2 |

---

## 2. User Experience Deep-Dive

### 2.1 The Operator's Mental Model

A real Astana transit operator thinks in these terms:

1. **"Which lines are overloaded RIGHT NOW?"** → Need live congestion heatmap on map
2. **"What will happen at 17:00 on Route 12?"** → Need route + time selector with forecast
3. **"Station X is closed — how do I reroute?"** → Need scenario with before/after comparison
4. **"Is my forecast accurate?"** → Need actual vs predicted overlay (requires real data input)
5. **"Send more buses to Line 5"** → Need capacity planning with actionable recommendations

### 2.2 Current UX Problems

**Streamlit (Research UI)**:
- Tab-heavy (7 tabs) — operators won't use this; it's for data scientists
- No way to input real ridership data for validation
- Forecast is per-station, not per-route — transit thinks in lines, not individual stops
- Drift viz shows affected stations but not business impact

**React Dashboard (Operator UI)**:
- LiveMap shows bus positions but no spatial intelligence (heatmap, clusters, route highlighting)
- CommandCenter shows KPIs but no drill-down — clicking a metric should navigate to detail
- Alerts are static mock data with no severity filtering or time-range
- ScenarioPlanner has no visualization of results
- Settings page exists but has no real configuration
- No dark mode, no responsive design for mobile/tablet

---

## 3. Improvement Plan

### Phase 1: Core Transit Intelligence (P0 — 2 weeks)

#### 3.1 Route & Time Selector on Map
- Add sidebar panel with route checkboxes (filter by line)
- Add time-of-day slider (0:00–23:59) that filters forecast data
- When route selected, highlight polyline, dim others
- Show route summary card: total ridership, avg load, peak hour

#### 3.2 Congestion Heatmap Overlay
- Overlay color-coded circles at each station on the map
- Color scale: green (<50% capacity) → yellow (50-80%) → red (>80%)
- Radius proportional to ridership volume
- Toggle button in map controls to show/hide heatmap
- Time-slider controls the hour shown

#### 3.3 Station Detail Panel
- Click station marker → slide-in panel showing:
  - Hourly ridership curve (last 24h)
  - Forecast next 6h with confidence band
  - Connected routes with their current load
  - Active alerts for this station
- New API: `GET /api/v1/stations/{id}/detail`

#### 3.4 Real Ridership Input (Streamlit)
- Upload CSV/Excel with columns: station, timestamp, passengers
- Validate against model's station list
- Show actual vs predicted overlay chart
- Compute MAE/RMSE on uploaded data vs model output

### Phase 2: Operational Intelligence (P1 — 2 weeks)

#### 3.5 Enhanced Bus Markers
- Show occupancy % as badge (green/yellow/red)
- Show speed (km/h), ETA to next stop
- Pulse animation for delayed buses
- Click bus → show route path + remaining stops

#### 3.6 Smart Alert Engine
- Auto-generate alerts from threshold rules:
  - Station >85% capacity for 30 min
  - Route avg load >90% during rush hour
  - Forecast spike >2x normal within 2h
- Alert severity: critical (>95%), warning (>80%), info (>60%)
- New endpoints: `GET /alerts/active`, `POST /alerts/rules`

#### 3.7 Scenario Before/After Comparison
- Show baseline ridership chart
- Overlay scenario result as dashed line
- Calculate: ridership delta %, capacity utilization change, affected stations count

#### 3.8 Forecast by Route
- Aggregate station-level forecasts into route-level
- Show route load over time (peak/off-peak)
- Streamlit: add route selector in Tab 5

### Phase 3: Polish & Completeness (P2 — 1 week)

#### 3.9 Timetable View
- New page: `/timetable` showing schedules per route
- Filter by route, direction, time range

#### 3.10 Reports Generation
- Daily operations summary
- Weekly ridership trends
- Exception report (stations over capacity)
- `GET /api/v1/reports/operations?date=YYYY-MM-DD&format=csv`

#### 3.11 Dark Mode & Responsive Design
- Tailwind dark mode toggle in Settings
- Responsive breakpoints for tablet (operators use iPads)

#### 3.12 Streamlit: Forecast by Line
- Add line-level forecast aggregation
- Show line load vs capacity chart
- "What if we add a bus?" scenario in Tab 5

---

## 4. Architecture Recommendations

### Backend API Gaps
```
GET  /api/v1/stations/{id}/detail     ← station detail with forecasts + alerts
GET  /api/v1/routes/{id}/forecast      ← route-level aggregated forecast
GET  /api/v1/routes/{id}/schedule      ← timetable data
POST /api/v1/ridership/upload          ← accept real ridership CSV
GET  /api/v1/alerts/active             ← active alerts with filtering
POST /api/v1/alerts/rules              ← configure alert thresholds
GET  /api/v1/reports/operations        ← operational reports
GET  /api/v1/heatmap?hour=17           ← congestion heatmap data
```

### Data Model Additions
- `ScheduleORM`: route_id, stop_id, arrival_offset_min, departure_offset_min
- `RidershipActualORM`: station_id, timestamp, passengers (real observed data)
- `AlertRuleORM`: metric, threshold, severity, route_id (nullable), station_id (nullable)

### Key Principle
**Transit operators think in ROUTES, not stations.** Every feature should have a route-first entry point. The current station-centric view is the data scientist's perspective — valuable but not the primary operator workflow.

---

## 5. Quick Wins (Can Do Today)

1. **Add route filter to LiveMap sidebar** — checkbox list of routes, highlight selected
2. **Color-code station markers by load** — green/yellow/red based on current ridership
3. **Add occupancy badge to bus markers** — show % in the marker tooltip
4. **Make alerts filterable by severity** — dropdown on AlertsPage
5. **Add "by route" tab to Operational Forecast** — aggregate station forecasts per line
