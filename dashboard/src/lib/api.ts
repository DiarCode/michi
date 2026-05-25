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
  RouteSchedule,
  OperationsReport,
} from "@/types";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1",
});

export const fetchStations = (hour?: number): Promise<{ stations: Station[] }> =>
  api.get("/stations", { params: hour !== undefined ? { hour } : {} }).then((r) => r.data);

export const fetchRoutes = (): Promise<{ routes: Route[] }> =>
  api.get("/routes").then((r) => r.data);

export const fetchRouteStops = (routeId: string): Promise<{ route_id: string; stops: RouteStop[] }> =>
  api.get(`/routes/${routeId}/stops`).then((r) => r.data);

export const fetchRouteForecast = (routeId: string): Promise<RouteForecast> =>
  api.get(`/routes/${routeId}/forecast`).then((r) => r.data);

export const fetchStationDetail = (stationId: string): Promise<StationDetail> =>
  api.get(`/stations/${stationId}/detail`).then((r) => r.data);

export const fetchKPIs = (): Promise<KPIData> =>
  api.get("/dashboard/kpis").then((r) => r.data);

export const fetchAlerts = (): Promise<{ alerts: Alert[] }> =>
  api.get("/alerts").then((r) => r.data);

export const fetchRouteSchedule = (routeId: string): Promise<RouteSchedule> =>
  api.get(`/routes/${routeId}/schedule`).then((r) => r.data);

export const fetchOperationsReport = (format?: string): Promise<OperationsReport> =>
  api.get("/dashboard/operations", { params: format ? { format } : {} }).then((r) => r.data);