import { useState } from "react";
import { useStations } from "@/hooks/useStations";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const DISTRICTS = ["All", "Esil", "Almaty", "Saryarka", "Baikonur", "Unknown"];

function loadColor(intensity: number): string {
  if (intensity > 0.7) return "bg-red-500 dark:bg-red-600";
  if (intensity > 0.4) return "bg-amber-400 dark:bg-amber-500";
  return "bg-emerald-400 dark:bg-emerald-500";
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
    <Card>
      <CardHeader className="flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm">Congestion Heatmap</CardTitle>
        <span className="text-xs text-gray-500 dark:text-gray-400">{filtered.length} stations</span>
      </CardHeader>
      <CardContent>
        <div className="flex gap-1 mb-3 flex-wrap">
          {DISTRICTS.map((d) => (
            <button key={d} onClick={() => setDistrict(d)}
              className={`px-2 py-1 text-xs rounded-md transition-colors ${district === d ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"}`}>
              {d}{d !== "All" && districtCounts[d] ? ` (${districtCounts[d]})` : ""}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-1.5 max-h-64 overflow-y-auto">
          {filtered.map((s) => {
            const intensity = (s.ridership_24h ?? 0) / maxRidership;
            return (
              <div key={s.id} className={`p-1.5 rounded text-[10px] text-center text-white ${loadColor(intensity)} truncate`} title={`${s.name}: ${(s.ridership_24h ?? 0).toLocaleString()}/day`}>
                {s.name.length > 12 ? s.name.slice(0, 10) + "…" : s.name}
              </div>
            );
          })}
        </div>
        <div className="flex items-center gap-4 mt-3 text-[10px] text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-emerald-400 dark:bg-emerald-500" /> &lt;40%</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-400 dark:bg-amber-500" /> 40–70%</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500 dark:bg-red-600" /> &gt;70%</span>
        </div>
      </CardContent>
    </Card>
  );
}
