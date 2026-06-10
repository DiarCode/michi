import { useQuery } from "@tanstack/react-query";
import { fetchRoutes } from "@/lib/api";
import { useRouteStore } from "@/stores/routeStore";
import { useEffect, useMemo } from "react";
import type { Route } from "@/types";

/**
 * Fetch the list of routes (5-minute stale time).
 */
export function useRoutes() {
  return useQuery({
    queryKey: ["routes"],
    queryFn: fetchRoutes,
    staleTime: 5 * 60 * 1000,
    select: (data) => data.routes,
  });
}

/**
 * Fetch route paths for the given route IDs and resolve station coordinates.
 * Returns a map of routeId → [lon, lat][] polyline coordinates.
 *
 * Call with the full list of routes once they're available, and this hook
 * will batch-fetch all route stop data and build polylines.
 */
export function useRoutePaths(routes: Route[], stationCoords: Record<string, { lat: number; lon: number }>) {
  const { routePaths, fetchRoutePaths, setStationLookup, fetchedRoutes } = useRouteStore();

  // Update the station coordinate lookup whenever station data changes
  useEffect(() => {
    setStationLookup(stationCoords);
  }, [stationCoords, setStationLookup]);

  // Build route color map for convenience
  const routeColorMap = useMemo(
    () => Object.fromEntries(routes.map((r) => [r.id, r.color ?? "#888"])),
    [routes],
  );

  // Fetch route paths for all routes on mount
  useEffect(() => {
    if (routes.length === 0) return;
    const unfetched = routes.filter((r) => !fetchedRoutes.has(r.id)).map((r) => r.id);
    if (unfetched.length > 0) {
      fetchRoutePaths(unfetched);
    }
  }, [routes, fetchedRoutes, fetchRoutePaths]);

  return { routePaths, routeColorMap };
}