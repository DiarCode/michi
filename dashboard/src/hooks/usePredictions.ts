import { useQuery } from "@tanstack/react-query";
import { fetchPredictions } from "@/lib/api";
import type { PredictionPoint } from "@/types";

/** Fetch predictions for a single horizon (in minutes) */
export function usePredictions(horizonMinutes?: number, enabled = true) {
  return useQuery({
    queryKey: ["predictions", horizonMinutes],
    queryFn: () => fetchPredictions(horizonMinutes),
    staleTime: 30 * 1000,
    enabled,
    select: (data) => data.predictions,
  });
}

/** Fetch predictions for multiple horizons simultaneously */
export function useAllHorizonPredictions(enabled = true) {
  const now = usePredictions(0, enabled);
  const h1 = usePredictions(60, enabled);
  const h2 = usePredictions(120, enabled);
  const h4 = usePredictions(240, enabled);

  const isLoading = now.isLoading || h1.isLoading || h2.isLoading || h4.isLoading;

  const byHorizon: Record<number, PredictionPoint[]> = {};
  if (now.data) byHorizon[0] = now.data;
  if (h1.data) byHorizon[60] = h1.data;
  if (h2.data) byHorizon[120] = h2.data;
  if (h4.data) byHorizon[240] = h4.data;

  return { predictions: byHorizon, isLoading };
}

/** Build a lookup map: station_id → PredictionPoint for a given horizon */
export function buildPredictionLookup(
  predictions: PredictionPoint[],
): Record<string, PredictionPoint> {
  const map: Record<string, PredictionPoint> = {};
  for (const p of predictions) {
    map[p.station_id] = p;
  }
  return map;
}