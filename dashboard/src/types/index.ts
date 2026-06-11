export interface Station {
  id: string
  name: string
  lat: number
  lon: number
  district?: string
  ridership_24h?: number
  load_percent?: number
  confidence_lower?: number
  confidence_upper?: number
}

export interface Route {
  id: string
  name: string
  color?: string
  stop_count?: number
  avg_ridership?: number
}

export interface RouteStop {
  id: string
  name: string
}

export interface Alert {
  id: number
  severity: string
  title: string
  message: string
  station_id?: string
  route_id?: string
  created_at: string
  acknowledged?: boolean
  auto?: boolean
  rule_id?: string
}

export interface RichAlert {
  id: number
  family?: string
  severity: "critical" | "warning" | "info"
  title: string
  what?: string
  when_hint?: string
  where_hint?: string
  why?: string
  confidence?: number
  recommended_actions?: ActionSuggestion[]
  consequence_if_ignored?: string
  sla_timer_minutes?: number
  acknowledged: boolean
  assigned_to?: string
  station_id?: string
  route_id?: string
  created_at: string
}

export interface ActionSuggestion {
  type: string
  label: string
  impact: { ridership_change: number; wait_time_change: number }
}

export interface BusPosition {
  bus_id: string
  route_id: string
  lat: number
  lon: number
  speed_kmh?: number
  next_stop?: string
  eta_seconds?: number
  occupancy_percent?: number
}

export interface ForecastPoint {
  station_id: string
  timestamp: string
  predicted: number
  confidence: number
  horizon_minutes?: number
}

export interface PredictionPoint {
  station_id: string
  timestamp: string
  predicted: number
  confidence: number
  horizon_minutes: number
  model_version: string
}

export interface KPIData {
  total_stations: number
  active_routes: number
  avg_ridership: number
  alerts_today: number
  on_time_performance?: number
  peak_hour?: string
}

export interface StationDetail {
  station: Station
  connected_routes: { id: string; name: string; color?: string }[]
  forecast: ForecastPoint[]
  alerts: { severity: string; title: string; message: string }[]
  hourly_ridership: { hour: number; ridership: number }[]
}

// Backend response wrappers
export interface StationListResponse {
  stations: Station[]
  hour?: number
}

export interface RouteListResponse {
  routes: Route[]
}

export interface RouteStopsResponse {
  route_id: string
  stops: RouteStop[]
}

export interface ForecastResponse {
  station_id: string
  forecast: ForecastPoint[]
}

export interface AlertListResponse {
  alerts: Alert[]
}

export interface ScenarioResult {
  scenario_id: string
  base_metrics: Record<string, number>
  scenario_metrics: Record<string, number>
  changes: Record<string, number>
}

export interface RouteForecast {
  route_id: string
  route: Route | null
  stop_count: number
  forecast: { hour: number; predicted: number; confidence: number }[]
  avg_ridership: number
}

export interface ScheduleEntry {
  stop_id: string
  stop_name: string
  time: string
  headway_min: number
  direction: string
}

export interface RouteSchedule {
  route_id: string
  route_name: string
  stops: RouteStop[]
  schedule: ScheduleEntry[]
  first_bus: string
  last_bus: string
  headway_min: number
}

export interface OperationsReport {
  date: string
  kpis: KPIData
  district_summary: Record<
    string,
    { stations: number; total_ridership: number }
  >
  peak_hours: string[]
  over_capacity_stations: { id: string; name: string; ridership_24h: number }[]
  total_stations: number
}

export interface AnalyticsSummary {
  ridership_by_district: Record<
    string,
    { total: number; avg_daily: number; peak_hour: number }
  >
  route_performance: {
    route_id: string
    name: string
    on_time_pct: number
    avg_wait_min: number
    daily_ridership: number
  }[]
  hourly_distribution: { hour: number; ridership: number }[]
}

export interface AnalyticsTrends {
  period_days: number
  trends: { date: string; ridership: number }[]
  avg_daily: number
  trend: string
  change_pct: number
}

export interface NetworkGraph {
  nodes: {
    id: string
    name: string
    lat: number
    lon: number
    district: string
  }[]
  edges: { from: string; to: string }[]
  districts: Record<string, number>
  stats: { total_stations: number; total_routes: number; total_edges: number }
}

export interface ForecastCompare {
  station_id: string | null
  models: {
    name: string
    mae: number
    rmse: number
    forecast: { hour: number; predicted: number }[]
  }[]
}

export interface TrainingStatus {
  status: string
  last_trained: string
  model_version: string
  metrics: { mae: number; rmse: number; mape: number }
  epochs_trained: number
  training_time_seconds: number
}

export interface Intervention {
  id: number
  alert_id?: number
  intervention_type: string
  route_id?: string
  station_id?: string
  created_at: string
  status: string
  operator_note?: string
  predicted_impact?: string
  actual_impact?: string
  approved_by?: string
}

export interface Suggestion {
  type: string
  priority: string
  title: string
  description: string
  station_id?: string
  route_ids?: string[]
  predicted_impact?: { ridership_change: number; wait_time_change: number }
  action?: string
  created_at: string
}

export interface ExecutiveKPIs {
  total_stations: number
  active_routes: number
  alerts_today: number
  critical_alerts: number
  interventions_today: number
  completed_interventions: number
  prediction_accuracy_mape?: number
  overcrowding_prevented: number
  on_time_performance: number
}

export interface ROISummary {
  total_investment: number
  annual_savings: number
  roi_pct: number
  payback_months: number
  breakdown: {
    category: string
    investment: number
    savings: number
  }[]
}

export interface DepotStatus {
  depots: {
    depot_id: string
    name: string
    lat: number
    lon: number
    total_buses: number
    available: number
    maintenance: number
    charging: number
    routes_served: string[]
  }[]
}

export interface PassengerCrowding {
  stations: {
    station_id: string
    name: string
    district?: string
    current_crowding: string
    predictions: {
      horizon_minutes: number
      predicted: number
      confidence: number
      level: string
    }[]
  }[]
}

export type UserRole =
  | "dispatch"
  | "research"
  | "planning"
  | "executive"
  | "depot"
  | "passenger"

// Timeline types
export type TimelineMode = "live" | "historical"
export type PlaybackSpeed = 1 | 2 | 5

export interface TimelinePoint {
  timestamp: string
  station_id: string
  actual: number | null
  predicted: number | null
  confidence_upper: number | null
  confidence_lower: number | null
}

export interface TimelineResponse {
  timeline: TimelinePoint[]
  resolution: string
  start_time: string
  end_time: string
  station_id: string | null
  total_points: number
}

// Simulation types
export interface SimulationTick {
  tick: number
  timestamp: string
  events: SimulationEvent[]
  metrics: ValidationMetric
  model_version: string
}

export interface SimulationEvent {
  type: string
  route_id?: string
  station_id?: string
  detail: string
}

export interface ValidationMetric {
  mae: number
  rmse?: number
  mape: number
  accuracy?: number
  drift_status?: "normal" | "warning" | "critical"
  tick?: number
  timestamp?: string
}

export interface DriftAlert {
  metric: string
  current_value: number
  baseline_value: number
  deviation_pct: number
  severity: "low" | "medium" | "high"
  timestamp: string
}

export interface SimulationState {
  running: boolean
  tick: number
  startTime: string | null
  metricsHistory: ValidationMetric[]
  driftAlerts: DriftAlert[]
  isStale: boolean
  lastTickAt: string | null
}

// Backend-aligned API response for GET /simulation/state
export interface SimulationStateResponse {
  running: boolean
  task_id: string | null
  tick: number
  current_time: string | null
  drift_status: "normal" | "warning" | "critical"
  metrics: {
    mae: number | null
    mape: number | null
    accuracy: number | null
  }
  station_count: number | null
}

// Connection status
export interface ConnectionStatus {
  connected: boolean
  lastTickReceived: number
  reconnectAttempt: number
  lastConnectedAt: string | null
}

// Weather types
export interface WeatherReading {
  id?: number
  timestamp: string
  temperature_c: number | null
  humidity_pct: number | null
  wind_speed_kmh: number | null
  precipitation_mm: number | null
  weather_code: number | null
  description: string | null
  is_forecast: boolean
  source?: string
}

export interface WeatherImpact {
  weather_code: number
  temperature_c: number | null
  description: string
  impact_factor: number
}

export interface ROISummary {
  total_interventions: number
  completed: number
  estimated_ridership_saved: number
  estimated_wait_time_saved_minutes: number
  fuel_savings_liters: number
  cost_per_intervention_usd: number
  total_cost_usd: number
  estimated_benefit_usd: number
  net_roi_pct: number
}

export interface SimulationTickData {
  tick: number
  current_time: string
  drift_status: string
  mae: number | null
  mape: number | null
  accuracy: number | null
}
