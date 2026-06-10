import { useMemo } from "react";
import { useBusStore } from "@/stores/busStore";
import { useRouteStore } from "@/stores/routeStore";
import { useStations } from "@/hooks/useStations";
import { computeConnectionRisks, riskSummary, type ConnectionRisk } from "@/lib/connectionProtection";

/**
 * Computes connection protection risks in real-time.
 * Recomputes every render when bus positions change.
 * Returns risks, summary counts, and critical risks.
 */
export function useConnectionProtection() {
  const busPositions = useBusStore((s) => s.buses);
  const stationToRoutes = useRouteStore((s) => s.stationToRoutes);
  const { data: stationData } = useStations();
  const stations = stationData?.stations ?? [];

  const buses = Object.values(busPositions);

  const risks: ConnectionRisk[] = useMemo(
    () => computeConnectionRisks(buses, stations, stationToRoutes),
    [buses, stations, stationToRoutes],
  );

  const summary = useMemo(() => riskSummary(risks), [risks]);
  const criticalRisks = useMemo(
    () => risks.filter((r) => r.riskLevel === "at_risk"),
    [risks],
  );

  return { risks, summary, criticalRisks };
}