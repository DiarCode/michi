import { useState } from "react";
import { useStations } from "@/hooks/useStations";
import { usePredictions, buildPredictionLookup } from "@/hooks/usePredictions";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const DISTRICTS = ["All", "Esil", "Almaty", "Saryarka", "Baikonur", "Unknown"];
const HORIZONS = [
  { minutes: 0, label: "Now" },
  { minutes: 60, label: "+1h" },
  { minutes: 120, label: "+2h" },
  { minutes: 240, label: "+4h" },
];

const STATION_CAPACITY = 3000;

function loadColor(intensity: number, predicted?: boolean): string {
  if (intensity > 0.8) return predicted ? "bg-destructive/70 text-white" : "bg-destructive text-white";
  if (intensity > 0.5) return predicted ? "bg-chart-4/70 text-white" : "bg-chart-4 text-white";
  return predicted ? "bg-chart-2/70 text-foreground" : "bg-chart-2 text-foreground";
}

export default function PredictiveHeatmap() {
  const { data } = useStations();
  const stations = data?.stations ?? [];
  const [district, setDistrict] = useState("All");
  const [horizon, setHorizon] = useState(0);

  // Fetch predictions for the selected horizon
  const { data: predictions = [] } = usePredictions(horizon > 0 ? horizon : undefined, true);
  const predMap = buildPredictionLookup(predictions);
  const isPredicted = horizon > 0;

  const filtered = district === "All" ? stations : stations.filter((s) => s.district === district);
  const districtCounts = stations.reduce<Record<string, number>>((acc, s) => {
    acc[s.district ?? "Unknown"] = (acc[s.district ?? "Unknown"] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <Card className="h-full">
      <CardHeader className="flex-row items-center justify-between pb-2">
        <div>
          <CardTitle>{isPredicted ? "Predictive Heatmap" : "Congestion Heatmap"}</CardTitle>
          {isPredicted && (
            <p className="text-xs text-muted-foreground mt-0.5">
              Showing forecast for +{horizon}min ahead
            </p>
          )}
        </div>
        <span className="text-sm text-muted-foreground font-medium">{filtered.length} stations</span>
      </CardHeader>
      <CardContent>
        {/* Time horizon selector */}
        <div className="flex gap-1.5 mb-3">
          {HORIZONS.map((h) => (
            <button
              key={h.minutes}
              onClick={() => setHorizon(h.minutes)}
              className={`px-3 py-1.5 text-xs rounded-full font-semibold transition-all ${
                horizon === h.minutes
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted text-muted-foreground border border-border hover:bg-border"
              }`}
            >
              {h.label}
            </button>
          ))}
        </div>

        {/* District filter */}
        <div className="flex gap-2 mb-4 flex-wrap">
          {DISTRICTS.map((d) => (
            <button
              key={d}
              onClick={() => setDistrict(d)}
              className={`px-3 py-1 text-xs rounded-full font-medium transition-all ${
                district === d
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted text-muted-foreground border border-border hover:bg-border"
              }`}
            >
              {d}{d !== "All" && districtCounts[d] ? ` (${districtCounts[d]})` : ""}
            </button>
          ))}
        </div>

        {/* Station grid */}
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 max-h-72 overflow-y-auto">
          {filtered.map((s) => {
            const pred = predMap[s.id];
            const ridership = pred ? pred.predicted : (s.ridership_24h ?? 0);
            const intensity = ridership / STATION_CAPACITY;
            const confidence = pred?.confidence;
            const label = pred ? `${Math.round(pred.predicted).toLocaleString()}` : `${(s.ridership_24h ?? 0).toLocaleString()}/day`;

            return (
              <div
                key={s.id}
                className={`px-2 py-2 rounded-xl text-xs text-center font-semibold ${loadColor(intensity, isPredicted)} truncate transition-colors duration-300`}
                title={`${s.name}: ${label}${confidence != null ? ` (${Math.round(confidence * 100)}% confidence)` : ""}`}
              >
                <div>{s.name.length > 12 ? s.name.slice(0, 10) + "…" : s.name}</div>
                {isPredicted && confidence != null && (
                  <div className="text-[10px] opacity-75">{Math.round(confidence * 100)}%</div>
                )}
              </div>
            );
          })}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-5 mt-4 text-xs text-muted-foreground font-medium">
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-chart-2" /> Low</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-chart-4" /> Medium</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-destructive" /> High</span>
          {isPredicted && (
            <span className="ml-auto text-chart-2">⚡ Predicted</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}