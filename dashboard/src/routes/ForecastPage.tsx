import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useStations } from "@/hooks/useStations";
import { fetchStationDetail } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

export default function ForecastPage() {
  const { data } = useStations();
  const stations = data?.stations ?? [];
  const [selectedStation, setSelectedStation] = useState<string>("");

  const { data: detail } = useQuery({
    queryKey: ["station-detail", selectedStation],
    queryFn: () => fetchStationDetail(selectedStation),
    enabled: !!selectedStation,
  });

  const forecast = detail?.forecast ?? [];
  const hourlyRidership = detail?.hourly_ridership ?? [];
  const maxVal = Math.max(...forecast.map((f) => f.predicted), ...hourlyRidership.map((h) => h.ridership), 1);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Station Forecast</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400">View hourly ridership forecasts for individual stations.</p>

      <Card>
        <CardHeader><CardTitle>Select Station</CardTitle></CardHeader>
        <CardContent>
          <select className="w-full border rounded px-3 py-2 dark:bg-gray-800" value={selectedStation} onChange={(e) => setSelectedStation(e.target.value)}>
            <option value="">— Choose a station —</option>
            {stations.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.district ?? "Unknown"})</option>)}
          </select>
        </CardContent>
      </Card>

      {detail && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Station</CardTitle>
              </CardHeader>
              <CardContent><p className="text-lg font-bold truncate">{detail.station.name}</p></CardContent>
            </Card>
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">District</CardTitle>
              </CardHeader>
              <CardContent><p className="text-lg font-bold">{detail.station.district ?? "—"}</p></CardContent>
            </Card>
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Forecast Points</CardTitle>
              </CardHeader>
              <CardContent><p className="text-lg font-bold">{forecast.length}</p></CardContent>
            </Card>
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Routes</CardTitle>
                <TrendingUp className="h-4 w-4 text-gray-400" />
              </CardHeader>
              <CardContent><p className="text-lg font-bold">{detail.connected_routes.length}</p></CardContent>
            </Card>
          </div>

          {detail.connected_routes.length > 0 && (
            <Card>
              <CardHeader><CardTitle>Connected Routes</CardTitle></CardHeader>
              <CardContent>
                <div className="flex gap-2 flex-wrap">
                  {detail.connected_routes.map((r) => (
                    <span key={r.id} className="px-3 py-1.5 rounded-full text-sm font-medium text-white" style={{ backgroundColor: r.color ?? "#3b82f6" }}>{r.name}</span>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {forecast.length > 0 && (
            <Card>
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle>Forecast vs Actual</CardTitle>
                <div className="flex gap-3 text-xs">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500" /> Forecast</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-emerald-400" /> Actual</span>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-end gap-1 h-48">
                  {forecast.map((f, i) => {
                    const actual = hourlyRidership.find((h) => h.hour === new Date(f.timestamp).getHours());
                    return (
                      <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
                        <div className="w-full flex flex-col gap-0.5" style={{ height: "160px" }}>
                          <div className="flex-1 flex items-end gap-0.5">
                            <div className="flex-1 bg-blue-500 rounded-t" style={{ height: `${(f.predicted / maxVal) * 100}%`, minHeight: "2px" }} />
                            {actual && <div className="flex-1 bg-emerald-400 rounded-t" style={{ height: `${(actual.ridership / maxVal) * 100}%`, minHeight: "2px" }} />}
                          </div>
                        </div>
                        {i % 3 === 0 && <span className="text-[9px] text-gray-400">{new Date(f.timestamp).getHours()}</span>}
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {detail.alerts.length > 0 && (
            <Card>
              <CardHeader><CardTitle className="text-amber-600">Active Alerts</CardTitle></CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {detail.alerts.map((a, i) => (
                    <div key={i} className="flex items-start gap-2 p-2 rounded bg-amber-50 dark:bg-amber-900/20">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${a.severity === "HIGH" ? "bg-red-100 text-red-700" : a.severity === "MEDIUM" ? "bg-amber-100 text-amber-700" : "bg-blue-100 text-blue-700"}`}>{a.severity}</span>
                      <div><p className="text-sm font-medium">{a.title}</p><p className="text-xs text-gray-500 dark:text-gray-400">{a.message}</p></div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}