export interface Station {
  id: string;
  name: string;
  lat: number;
  lon: number;
  district?: string;
  ridership_24h?: number;
  load_percent?: number;
}

export interface Route {
  id: string;
  name: string;
  color?: string;
  stop_count?: number;
  avg_ridership?: number;
}

export interface RouteStop {
  id: string;
  name: string;
}

export interface Alert {
  id: number;
  severity: string;
  title: string;
  message: string;
  station_id?: string;
  route_id?: string;
  created_at: string;
}

export interface RichAlert {
  id: number;
  family?: string;
  severity: "critical" | "warning" | "info";
  title: string;
  what?: string;
  when_hint?: string;
  where_hint?: string;
  why?: string;
  confidence?: number;
  recommended_actions?: ActionSuggestion[];
  consequence_if_ignored?: string;
  sla_timer_minutes?: number;
  acknowledged: boolean;
  assigned_to?: string;
  station_id?: string;
  route_id?: string;
  created_at: string;
}

export interface ActionSuggestion {
  type: string;
  label: string;
  impact: { ridership_change: number; wait_time_change: number };
}

export interface BusPosition {
  bus_id: string;
  route_id: string;
  lat: number;
  lon: number;
  speed_kmh?: number;
  next_stop?: string;
  eta_seconds?: number;
  occupancy_percent?: number;
}

export interface ForecastPoint {
  station_id: string;
  timestamp: string;
  predicted: number;
  confidence: number;
  horizon_minutes?: number;
}

export interface PredictionPoint {
  station_id: string;
  timestamp: string;
  predicted: number;
  confidence: number;
  horizon_minutes: number;
  model_version: string;
}

export interface KPIData {
  total_stations: number;
  active_routes: number;
  avg_ridership: number;
  alerts_today: number;
  on_time_performance?: number;
  peak_hour?: string;
}

export interface StationDetail {
  station: Station;
  connected_routes: { id: string; name: string; color?: string }[];
  forecast: ForecastPoint[];
  alerts: { severity: string; title: string; message: string }[];
  hourly_ridership: { hour: number; ridership: number }[];
}

export interface RouteForecast {
  route_id: string;
  route: Route | null;
  stop_count: number;
  forecast: { hour: number; predicted: number; confidence: number }[];
  avg_ridership: number;
}

export interface ScheduleEntry {
  stop_id: string;
  stop_name: string;
  time: string;
  headway_min: number;
  direction: string;
}

export interface RouteSchedule {
  route_id: string;
  route_name: string;
  stops: RouteStop[];
  schedule: ScheduleEntry[];
  first_bus: string;
  last_bus: string;
  headway_min: number;
}

export interface OperationsReport {
  date: string;
  kpis: KPIData;
  district_summary: Record<string, { stations: number; total_ridership: number }>;
  peak_hours: string[];
  over_capacity_stations: { id: string; name: string; ridership_24h: number }[];
  total_stations: number;
}

export interface AnalyticsSummary {
  ridership_by_district: Record<string, { total: number; avg_daily: number; peak_hour: number }>;
  route_performance: { route_id: string; name: string; on_time_pct: number; avg_wait_min: number; daily_ridership: number }[];
  hourly_distribution: { hour: number; ridership: number }[];
}

export interface AnalyticsTrends {
  period_days: number;
  trends: { date: string; ridership: number }[];
  avg_daily: number;
  trend: string;
  change_pct: number;
}

export interface NetworkGraph {
  nodes: { id: string; name: string; lat: number; lon: number; district: string }[];
  edges: { from: string; to: string }[];
  districts: Record<string, number>;
  stats: { total_stations: number; total_routes: number; total_edges: number };
}

export interface ForecastCompare {
  station_id: string | null;
  models: { name: string; mae: number; rmse: number; forecast: { hour: number; predicted: number }[] }[];
}

export interface TrainingStatus {
  status: string;
  last_trained: string;
  model_version: string;
  metrics: { mae: number; rmse: number; mape: number };
  epochs_trained: number;
  training_time_seconds: number;
}

export interface Intervention {
  id: number;
  alert_id?: number;
  intervention_type: string;
  route_id?: string;
  station_id?: string;
  created_at: string;
  status: string;
  operator_note?: string;
  predicted_impact?: string;
  actual_impact?: string;
  approved_by?: string;
}

export interface Suggestion {
  type: string;
  priority: string;
  title: string;
  description: string;
  station_id?: string;
  route_ids?: string[];
  predicted_impact?: { ridership_change: number; wait_time_change: number };
  action?: string;
  created_at: string;
}

export interface ExecutiveKPIs {
  total_stations: number;
  active_routes: number;
  alerts_today: number;
  critical_alerts: number;
  interventions_today: number;
  completed_interventions: number;
  prediction_accuracy_mape?: number;
  overcrowding_prevented: number;
  on_time_performance: number;
}

export interface DepotStatus {
  depots: {
    depot_id: string;
    name: string;
    lat: number;
    lon: number;
    total_buses: number;
    available: number;
    maintenance: number;
    charging: number;
    routes_served: string[];
  }[];
}

export interface PassengerCrowding {
  stations: {
    station_id: string;
    name: string;
    district?: string;
    current_crowding: string;
    predictions: { horizon_minutes: number; predicted: number; confidence: number; level: string }[];
  }[];
}

export type UserRole =
  | "dispatch"
  | "research"
  | "planning"
  | "executive"
  | "depot"
  | "passenger";
