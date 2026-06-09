import { useState } from "react";
import { useStations } from "@/hooks/useStations";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const DISTRICTS = ["All", "Esil", "Almaty", "Saryarka", "Baikonur", "Unknown"];

function loadColor(intensity: number): string {
  if (intensity > 0.7) return "bg-michi-red text-white";
  if (intensity > 0.4) return "bg-michi-amber text-white";
  return "bg-michi-lime text-michi-dark";
}

export default function CongestionHeatmap() {
  const { data } = useStations();
  const stations = data?.stations ?? [];
  const [district, setDistrict] = useState("All");
  const maxRidership = Math.max(...stations.map((s) => s.ridership_24h ?? 0), 1);

  const filtered = district === "All" ? stations : stations.filter((s) => s.district === district);
  const districtCounts = stations.reduce<Record<string, number>>((acc, s) => {
    acc[s.district ?? "Unknown"] = (acc[s.district ?? "Unknown"] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <Card className="h-full">
      <CardHeader className="flex-row items-center justify-between pb-2">
        <CardTitle>Congestion Heatmap</CardTitle>
        <span className="text-sm text-michi-muted font-medium">{filtered.length} stations</span>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2 mb-4 flex-wrap">
          {DISTRICTS.map((d) => (
            <button key={d} onClick={() => setDistrict(d)}
              className={`px-3.5 py-1.5 text-xs rounded-full font-semibold transition-all ${
                district === d
                  ? "bg-michi-dark text-white shadow-sm"
                  : "bg-michi-warm text-michi-body border border-michi-border hover:bg-michi-border"
              }`}>
              {d}{d !== "All" && districtCounts[d] ? ` (${districtCounts[d]})` : ""}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 max-h-72 overflow-y-auto">
          {filtered.map((s) => {
            const intensity = (s.ridership_24h ?? 0) / maxRidership;
            return (
              <div key={s.id} className={`px-2 py-2 rounded-xl text-xs text-center font-semibold ${loadColor(intensity)} truncate`} title={`${s.name}: ${(s.ridership_24h ?? 0).toLocaleString()}/day`}>
                {s.name.length > 12 ? s.name.slice(0, 10) + "…" : s.name}
              </div>
            );
          })}
        </div>
        <div className="flex items-center gap-5 mt-4 text-xs text-michi-muted font-medium">
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-michi-lime" /> Low</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-michi-amber" /> Medium</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-michi-red" /> High</span>
        </div>
      </CardContent>
    </Card>
  );
}