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
