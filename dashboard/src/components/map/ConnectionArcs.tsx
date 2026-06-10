import { MapArc } from "@/components/ui/map";
import type { MapArcDatum } from "@/components/ui/map";
import { useConnectionProtection } from "@/hooks/useConnectionProtection";

interface ConnectionArcDatum extends MapArcDatum {
  riskLevel: "at_risk" | "tight";
}

/**
 * Renders arc overlays on the map for at-risk and tight connections.
 * For each connection risk, draws two arcs:
 * 1. Arriving bus → station (incoming)
 * 2. Station → departing bus (outgoing)
 */
export default function ConnectionArcs() {
  const { risks } = useConnectionProtection();

  // Only show arcs for at_risk and tight connections
  const visibleRisks = risks.filter((r) => r.riskLevel !== "safe");
  if (visibleRisks.length === 0) return null;

  // Build arc data: arriving→station and station→departing for each risk
  const arcs: ConnectionArcDatum[] = [];

  for (const r of visibleRisks.slice(0, 8)) {
    // Arc from arriving bus to station
    arcs.push({
      id: `conn-arr-${r.arriving.busId}-${r.departing.busId}-${r.stationId}`,
      from: r.arriving.coords,
      to: r.stationCoords,
      riskLevel: r.riskLevel as "at_risk" | "tight",
    });

    // Arc from station to departing bus
    arcs.push({
      id: `conn-dep-${r.arriving.busId}-${r.departing.busId}-${r.stationId}`,
      from: r.stationCoords,
      to: r.departing.coords,
      riskLevel: r.riskLevel as "at_risk" | "tight",
    });
  }

  // Group arcs by risk level for separate MapArc layers with distinct styling
  const atRiskArcs = arcs.filter((a) => a.riskLevel === "at_risk");
  const tightArcs = arcs.filter((a) => a.riskLevel === "tight");

  return (
    <>
      {atRiskArcs.length > 0 && (
        <MapArc
          data={atRiskArcs}
          id="conn-arcs-atrisk"
          curvature={0.2}
          paint={{
            "line-color": "var(--destructive)",
            "line-width": 3,
            "line-opacity": 0.8,
            "line-dasharray": [2, 2],
          }}
        />
      )}
      {tightArcs.length > 0 && (
        <MapArc
          data={tightArcs}
          id="conn-arcs-tight"
          curvature={0.2}
          paint={{
            "line-color": "var(--chart-4)",
            "line-width": 2,
            "line-opacity": 0.6,
            "line-dasharray": [4, 2],
          }}
        />
      )}
    </>
  );
}