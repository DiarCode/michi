import { create } from "zustand";
import { fetchRouteStops } from "@/lib/api";
import type { RouteStop } from "@/types";

interface RouteStoreState {
  /** Route ID → ordered list of stops */
  routeStops: Record<string, RouteStop[]>;
  /** Route ID → ordered [longitude, latitude] polyline coordinates */
  routePaths: Record<string, [number, number][]>;
  /** Station ID → Set of route IDs that include this station */
  stationToRoutes: Record<string, Set<string>>;
  /** Which routes have been fetched */
  fetchedRoutes: Set<string>;
  /** Loading state per route */
  loading: Record<string, boolean>;
  fetchRoutePath: (routeId: string) => Promise<[number, number][] | undefined>;
  /** Batch fetch multiple route paths */
  fetchRoutePaths: (routeIds: string[]) => Promise<Record<string, [number, number][]>>;
  /** Check if a station belongs to a route */
  isStationOnRoute: (stationId: string, routeId: string) => boolean;
  /** Resolve station coordinates from a station lookup */
  setStationLookup: (lookup: Record<string, { lat: number; lon: number }>) => void;
}

/** Station coordinates lookup — set externally when station data loads */
let stationCoords: Record<string, { lat: number; lon: number }> = {};

export const useRouteStore = create<RouteStoreState>((set, get) => ({
  routeStops: {},
  routePaths: {},
  stationToRoutes: {},
  fetchedRoutes: new Set(),
  loading: {},

  setStationLookup: (lookup) => {
    stationCoords = lookup;
  },

  fetchRoutePath: async (routeId: string) => {
    const state = get();
    if (state.fetchedRoutes.has(routeId)) {
      return state.routePaths[routeId];
    }
    if (state.loading[routeId]) return undefined;

    set((s) => ({ loading: { ...s.loading, [routeId]: true } }));

    try {
      const data = await fetchRouteStops(routeId);
      const stops = data.stops;

      // Build polyline by resolving stop IDs/names to station coordinates
      const path: [number, number][] = [];
      for (const stop of stops) {
        const coords = stationCoords[stop.id] ?? stationCoords[stop.name];
        if (coords) {
          path.push([coords.lon, coords.lat]);
        }
      }

      // Build station→route membership
      const stationToRoutes = { ...state.stationToRoutes };
      for (const stop of stops) {
        const key = stop.id || stop.name;
        if (!stationToRoutes[key]) {
          stationToRoutes[key] = new Set();
        }
        stationToRoutes[key].add(routeId);
      }

      set((s) => ({
        routeStops: { ...s.routeStops, [routeId]: stops },
        routePaths: { ...s.routePaths, [routeId]: path },
        stationToRoutes,
        fetchedRoutes: new Set([...s.fetchedRoutes, routeId]),
        loading: { ...s.loading, [routeId]: false },
      }));

      return path;
    } catch {
      set((s) => ({ loading: { ...s.loading, [routeId]: false } }));
      return undefined;
    }
  },

  fetchRoutePaths: async (routeIds: string[]) => {
    const results: Record<string, [number, number][]> = {};
    await Promise.all(
      routeIds.map(async (id) => {
        const path = await get().fetchRoutePath(id);
        if (path) results[id] = path;
      }),
    );
    return results;
  },

  isStationOnRoute: (stationId: string, routeId: string) => {
    const membership = get().stationToRoutes[stationId];
    return membership?.has(routeId) ?? false;
  },
}));