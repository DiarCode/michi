export interface Station {
  id: number;
  stop_id: string;
  name: string;
  lat: number;
  lon: number;
  district?: string;
}

export interface Route {
  id: number;
  route_id: string;
  name: string;
  color?: string;
  stop_sequence: number[];
}

export interface Alert {
  id: number;
  alert_type: string;
  severity: string;
  message: string;
  created_at: string;
}

export interface BusPosition {
  bus_id: string;
  route_id: string;
  lat: number;
  lon: number;
  speed_kmh: number;
  next_stop: string;
  eta_seconds: number;
  occupancy_percent: number;
}
