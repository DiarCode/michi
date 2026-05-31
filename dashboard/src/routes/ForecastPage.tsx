import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useStations } from "@/hooks/useStations";
import { fetchStationDetail } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";
import ForecastChart from "@/components/dashboard/ForecastChart";
import { CardSkeleton } from "@/components/ui/skeleton";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";

export default function ForecastPage() {
  const { data, isLoading: loadingStations } = useStations();
  const stations = data?.stations ?? [];
  const [selectedStation, setSelectedStation] = useState<string>("");

  const { data: detail } = useQuery({
    queryKey: ["station-detail", selectedStation],
    queryFn: () => fetchStationDetail(selectedStation),
    enabled: !!selectedStation,
  });

  if (loadingStations) return <div className="p-6 space-y-6"><CardSkeleton /><CardSkeleton /></div>;

  const forecast = detail?.forecast ?? [];
  const hourlyRidership = detail?.hourly_ridership ?? [];
  const maxVal = Math.max(...forecast.map((f) => f.predicted), ...hourlyRidership.map((h) => h.ridership), 1);

  const comparisonData = forecast.map((f) => {
    const actual = hourlyRidership.find((h) => h.hour === new Date(f.timestamp).getHours());
    return {
      hour: new Date(f.timestamp).getHours(),
      predicted: f.predicted,
      actual: actual?.ridership ?? null,
    };
  });

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

          {selectedStation && (
            <ForecastChart stationId={selectedStation} stationName={detail.station.name} />
          )}

          {comparisonData.length > 0 && (
            <Card>
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle>Forecast vs Actual</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} domain={[0, maxVal]} />
                    <Tooltip
                      formatter={(value: unknown, name: string) => [
                        typeof value === "number" ? `${value} pax` : "N/A",
                        name === "predicted" ? "Forecast" : "Actual",
                      ]}
                      labelFormatter={(label: number) => `${String(label).padStart(2, "0")}:00`}
                    />
                    <Legend formatter={(value: string) => (value === "predicted" ? "Forecast" : "Actual")} />
                    <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="actual" stroke="#34d399" strokeWidth={2} dot={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
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