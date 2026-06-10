export type UserRole = "dispatch" | "research" | "planning" | "executive" | "superadmin"

export interface Station {
  id: string
  name: string
  district?: string
  lat?: number
  lon?: number
  ridership_24h?: number
}

export interface Route {
  id: string
  name: string
  short_name?: string
  color?: string
}

export interface RouteStop {
  station_id: string
  station_name: string
  sequence: number
  arrival_offset_minutes?: number
}

export interface RouteForecast {
  route_id: string
  forecast: ForecastPoint[]
}

export interface ForecastPoint {
  timestamp: string
  predicted: number
  confidence: number
}

export interface StationDetail extends Station {
  routes?: RouteStop[]
  forecast?: ForecastPoint[]
  current_load?: number
}

export interface KPIData {
  total_stations?: number
  active_routes?: number
  avg_ridership?: number
  on_time_performance?: number
  alerts_today?: number
  peak_hour?: string
}

export interface Alert {
  id: number
  title: string
  severity: "critical" | "warning" | "info"
  message: string
  created_at: string
  acknowledged?: boolean
  route_id?: string
  station_id?: string
}

export interface RichAlert extends Alert {
  family?: "station" | "route" | "forecast" | "system"
  why?: string
  confidence?: number
  consequence_if_ignored?: string
  recommended_actions?: { type: string; label: string }[]
  sla_timer_minutes?: number
}

export interface RouteSchedule {
  route_id: string
  schedule: { departure: string; station: string }[]
}

export interface OperationsReport {
  generated_at: string
  summary: Record<string, unknown>
}

export interface AnalyticsSummary {
  total_passengers: number
  total_alerts: number
  avg_accuracy: number
  network_health: number
}

export interface AnalyticsTrends {
  days: number
  series: { date: string; passengers: number; accuracy: number; alerts: number }[]
}

export interface NetworkGraph {
  nodes: { id: string; name: string }[]
  edges: { source: string; target: string; weight: number }[]
}

export interface ForecastCompare {
  stations: { station_id: string; station_name: string; predicted: number; actual?: number }[]
}

export interface TrainingStatus {
  running: boolean
  epoch: number
  total_epochs: number
  loss: number
  val_loss: number
  accuracy: number
}

export interface Intervention {
  id: number
  alert_id?: number
  intervention_type: string
  status: "pending" | "approved" | "executing" | "completed" | "cancelled"
  route_id?: string
  station_id?: string
  created_at: string
}

export interface Suggestion {
  id?: string
  priority: "low" | "medium" | "high" | "critical"
  type: string
  title: string
  description: string
  action?: string
}

export interface ExecutiveKPIs {
  ridership_total: number
  ridership_delta: number
  on_time: number
  on_time_delta: number
  cost_per_ride: number
  cost_per_ride_delta: number
  customer_satisfaction: number
  customer_satisfaction_delta: number
}

export interface DepotStatus {
  depot_id: string
  buses_ready: number
  buses_in_service: number
  buses_maintenance: number
}

export interface PassengerCrowding {
  timestamp: string
  routes: { route_id: string; crowding_pct: number }[]
}

export interface PredictionPoint {
  station_id: string
  timestamp: string
  predicted: number
  confidence: number
}

export interface TimelineResponse {
  station_id: string
  start_time: string
  end_time: string
  points: { timestamp: string; value: number; predicted?: number }[]
}
