/// <reference types="vite/client" />
import axios from "axios";
import type {
  Station,
  Route,
  RouteStop,
  RouteForecast,
  StationDetail,
  KPIData,
  Alert,
  RichAlert,
  RouteSchedule,
  OperationsReport,
  AnalyticsSummary,
  AnalyticsTrends,
  NetworkGraph,
  ForecastCompare,
  TrainingStatus,
  Intervention,
  Suggestion,
  ExecutiveKPIs,
  DepotStatus,
  PassengerCrowding,
  PredictionPoint,
  TimelineResponse,
} from "@/types";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1",
});

// Stations
export const fetchStations = (hour?: number): Promise<{ stations: Station[] }> =>
  api.get("/stations", { params: hour !== undefined ? { hour } : {} }).then((r) => r.data);

export const fetchStationDetail = (stationId: string): Promise<StationDetail> =>
  api.get(`/stations/${stationId}/detail`).then((r) => r.data);

// Routes
export const fetchRoutes = (): Promise<{ routes: Route[] }> =>
  api.get("/routes").then((r) => r.data);

export const fetchRouteStops = (routeId: string): Promise<{ route_id: string; stops: RouteStop[] }> =>
  api.get(`/routes/${routeId}/stops`).then((r) => r.data);

export const fetchRouteForecast = (routeId: string): Promise<RouteForecast> =>
  api.get(`/routes/${routeId}/forecast`).then((r) => r.data);

export const fetchRouteSchedule = (routeId: string): Promise<RouteSchedule> =>
  api.get(`/routes/${routeId}/schedule`).then((r) => r.data);

// Dashboard
export const fetchKPIs = (): Promise<KPIData> =>
  api.get("/dashboard/kpis").then((r) => r.data);

export const fetchOperationsReport = (format?: string): Promise<OperationsReport> =>
  api.get("/dashboard/operations", { params: format ? { format } : {} }).then((r) => r.data);

export const fetchSuggestions = (): Promise<{ suggestions: Suggestion[] }> =>
  api.get("/dashboard/suggestions").then((r) => r.data);

// Alerts
export const fetchAlerts = (): Promise<{ alerts: Alert[] }> =>
  api.get("/alerts").then((r) => r.data);

export const fetchRichAlerts = (): Promise<{ alerts: RichAlert[] }> =>
  api.get("/alerts/rich").then((r) => r.data).catch(() => fetchAlerts());

export const ackAlert = (alertId: number): Promise<{ success: boolean }> =>
  api.post(`/alerts/${alertId}/ack`).then((r) => r.data);

// Scenarios
export const runScenario = (name: string, modifications: unknown[]): Promise<Intervention> =>
  api.post("/scenarios/run", { name, modifications }).then((r) => r.data);

// Analytics
export const fetchAnalyticsSummary = (): Promise<AnalyticsSummary> =>
  api.get("/analytics/summary").then((r) => r.data);

export const fetchAnalyticsTrends = (days?: number): Promise<AnalyticsTrends> =>
  api.get("/analytics/trends", { params: days ? { days } : {} }).then((r) => r.data);

export const fetchNetworkGraph = (): Promise<NetworkGraph> =>
  api.get("/analytics/graph").then((r) => r.data);

export const fetchForecastCompare = (stationId?: string): Promise<ForecastCompare> =>
  api.get("/analytics/compare", { params: stationId ? { station_id: stationId } : {} }).then((r) => r.data);

export const fetchTrainingStatus = (): Promise<TrainingStatus> =>
  api.get("/analytics/status").then((r) => r.data);

export const startTraining = (epochs?: number): Promise<{ status: string; epochs: number; model_version: string; estimated_time_seconds: number }> =>
  api.post("/analytics/start", null, { params: epochs ? { epochs } : {} }).then((r) => r.data);

export const uploadRidership = (file: File): Promise<{ status: string; rows_received: number; filename: string }> => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/analytics/upload", form).then((r) => r.data);
};

// Predictions
export const fetchPredictions = (horizonMinutes?: number): Promise<{ predictions: PredictionPoint[] }> =>
  api.get("/analytics/predictions", { params: horizonMinutes ? { horizon_minutes: horizonMinutes } : {} }).then((r) => r.data);

// Interventions
export const fetchInterventions = (status?: string): Promise<{ interventions: Intervention[] }> =>
  api.get("/interventions", { params: status ? { status } : {} }).then((r) => r.data);

export const createIntervention = (params: { alert_id?: number; intervention_type: string; route_id?: string; station_id?: string }): Promise<Intervention> =>
  api.post("/interventions", null, { params }).then((r) => r.data);

export const simulateIntervention = (type: string, routeId?: string, stationId?: string): Promise<Record<string, unknown>> =>
  api.get("/interventions/simulate", { params: { intervention_type: type, route_id: routeId, station_id: stationId } }).then((r) => r.data);

export const updateInterventionStatus = (id: number, status: string, approvedBy?: string): Promise<Intervention> =>
  api.patch(`/interventions/${id}`, null, { params: { status, approved_by: approvedBy } }).then((r) => r.data);

// Executive
export const fetchExecutiveKPIs = (): Promise<ExecutiveKPIs> =>
  api.get("/executive/kpis").then((r) => r.data);

export const fetchExecutiveTrends = (days?: number): Promise<AnalyticsTrends> =>
  api.get("/executive/trends", { params: days ? { days } : {} }).then((r) => r.data);

export const fetchROISummary = (): Promise<Record<string, unknown>> =>
  api.get("/executive/roi").then((r) => r.data);

// Depot
export const fetchDepotStatus = (): Promise<DepotStatus> =>
  api.get("/depot/status").then((r) => r.data);

export const fetchDepotRecommendations = (depotId: string): Promise<{ depot_id: string; recommendations: unknown[] }> =>
  api.get(`/depot/${depotId}/dispatch-recommendations`).then((r) => r.data);

// Passenger Info
export const fetchPassengerCrowding = (): Promise<PassengerCrowding> =>
  api.get("/passenger/crowding").then((r) => r.data);

export const fetchServiceChanges = (): Promise<{ service_changes: unknown[] }> =>
  api.get("/passenger/service-changes").then((r) => r.data);

export const fetchMessagingTemplates = (): Promise<{ templates: unknown[] }> =>
  api.get("/passenger/messaging-templates").then((r) => r.data);

// Timeline
export const fetchTimeline = (params: {
  station_id?: string;
  start_time: string;
  end_time: string;
  resolution?: string;
}): Promise<TimelineResponse> =>
  api.get("/timeline", { params }).then((r) => r.data);

// Simulation
export const startSimulation = (): Promise<{ status: string; task_id?: string }> =>
  api.post("/simulation/start").then((r) => r.data);

export const stopSimulation = (): Promise<{ status: string; task_id?: string }> =>
  api.post("/simulation/stop").then((r) => r.data);

export const fetchSimulationState = (): Promise<{
  running: boolean;
  task_id?: string;
  tick: number;
  current_time?: string;
  drift_status: string;
  metrics: { mae: number | null; mape: number | null; accuracy: number | null };
  station_count?: number;
}> => api.get("/simulation/state").then((r) => r.data);

export const fetchSimulationMetrics = (hoursBack?: number): Promise<{
  realtime: { tick: number; mae: number; mape: number; accuracy: number; drift_status: string; timestamp: string }[];
  database: { timestamp: string; mae: number | null; mape: number | null; count: number }[];
  hours_back: number;
}> => api.get("/simulation/metrics", { params: hoursBack ? { hours_back: hoursBack } : {} }).then((r) => r.data);
