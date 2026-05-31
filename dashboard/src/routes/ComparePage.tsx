import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useStations } from "@/hooks/useStations";
import { fetchForecastCompare } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ChartSkeleton } from "@/components/ui/skeleton";

const MODEL_COLORS: Record<string, string> = {
  "DTS-GSSF": "#3b82f6",
  LSTM: "#ef4444",
  GRU: "#f59e0b",
  Transformer: "#8b5cf6",
  "Seasonal Naive": "#6b7280",
};

export default function ComparePage() {
  const { data } = useStations();
  const stations = data?.stations ?? [];
  const [stationId, setStationId] = useState<string>("");

  const { data: compare, isLoading } = useQuery({
    queryKey: ["forecast-compare", stationId],
    queryFn: () => fetchForecastCompare(stationId || undefined),
    enabled: true,
  });

  if (isLoading) return <div className="p-6 space-y-6"><ChartSkeleton /></div>;

  const models = compare?.models ?? [];
  const hours = models.length > 0 ? models[0].forecast.map((f) => f.hour) : [];
  const maxPredicted = Math.max(...models.flatMap((m) => m.forecast.map((f) => f.predicted)), 1);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Model Comparison</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400">Compare DTS-GSSF forecasts against baseline models.</p>

      <Card>
        <CardHeader><CardTitle>Filter by Station</CardTitle></CardHeader>
        <CardContent>
          <select className="w-full border rounded px-3 py-2 dark:bg-gray-800" value={stationId} onChange={(e) => setStationId(e.target.value)}>
            <option value="">All stations (average)</option>
            {stations.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </CardContent>
      </Card>

      {models.length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader><CardTitle>Error Metrics</CardTitle></CardHeader>
              <CardContent>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Model</th>
                      <th className="text-right py-2">MAE</th>
                      <th className="text-right py-2">RMSE</th>
                      <th className="text-right py-2">Rank</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...models].sort((a, b) => a.mae - b.mae).map((m, i) => (
                      <tr key={m.name} className="border-b hover:bg-gray-50 dark:hover:bg-gray-800">
                        <td className="py-2 font-medium flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: MODEL_COLORS[m.name] ?? "#9ca3af" }} />
                          {m.name}
                        </td>
                        <td className="text-right py-2 font-mono">{m.mae}</td>
                        <td className="text-right py-2 font-mono">{m.rmse}</td>
                        <td className="text-right py-2">
                          <span className={`px-2 py-0.5 rounded text-xs font-bold ${i === 0 ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300" : "bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300"}`}>
                            #{i + 1}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Model Forecast Comparison</CardTitle></CardHeader>
              <CardContent>
                <div className="flex items-end gap-0.5 h-48">
                  {hours.map((h) => (
                    <div key={h} className="flex-1 flex flex-col items-center gap-0">
                      <div className="w-full flex items-end gap-px" style={{ height: "160px" }}>
                        {models.map((m) => {
                          const point = m.forecast.find((f) => f.hour === h);
                          if (!point) return null;
                          return (
                            <div key={m.name} className="flex-1 rounded-t" style={{ backgroundColor: MODEL_COLORS[m.name] ?? "#9ca3af", height: `${(point.predicted / maxPredicted) * 100}%`, minHeight: "1px" }} title={`${m.name} ${h}:00 — ${point.predicted}`} />
                          );
                        })}
                      </div>
                      {h % 3 === 0 && <span className="text-[9px] text-gray-400">{h}</span>}
                    </div>
                  ))}
                </div>
                <div className="flex gap-4 mt-3 flex-wrap text-xs">
                  {models.map((m) => (
                    <span key={m.name} className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded-full" style={{ backgroundColor: MODEL_COLORS[m.name] ?? "#9ca3af" }} />
                      {m.name}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader><CardTitle>Hourly Predictions by Model</CardTitle></CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-1.5 pr-3">Hour</th>
                      {models.map((m) => (
                        <th key={m.name} className="text-right py-1.5 px-2">{m.name}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {hours.map((h) => (
                      <tr key={h} className="border-b hover:bg-gray-50 dark:hover:bg-gray-800">
                        <td className="py-1.5 font-mono">{String(h).padStart(2, "0")}:00</td>
                        {models.map((m) => {
                          const point = m.forecast.find((f) => f.hour === h);
                          return (
                            <td key={m.name} className="text-right py-1.5 px-2 font-mono">
                              {point?.predicted.toLocaleString() ?? "—"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}