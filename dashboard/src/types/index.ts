export interface Station {
  id: string;
  name: string;
  lat: number;
  lon: number;
  district?: string;
  ridership_24h?: number;
}

export interface Route {
  id: string;
  name: string;
  color?: string;
  stop_count?: number;
  avg_ridership?: number;
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
