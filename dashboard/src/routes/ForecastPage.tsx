import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useStations } from "@/hooks/useStations";
import { fetchStationDetail, fetchPredictions } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  TrendingUp, Users, Target, BarChart3, AlertTriangle, CheckCircle2,
} from "lucide-react";
import {
  Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Area, AreaChart,
} from "recharts";
import { CardSkeleton } from "@/components/ui/skeleton";

const HORIZONS = [
  { label: "15 min", minutes: 15 },
  { label: "30 min", minutes: 30 },
  { label: "1 hour", minutes: 60 },
  { label: "2 hours", minutes: 120 },
  { label: "6 hours", minutes: 360 },
  { label: "12 hours", minutes: 720 },
  { label: "24 hours", minutes: 1440 },
] as const;

export default function ForecastPage() {
  const { data, isLoading: loadingStations } = useStations();
  const stations = data?.stations ?? [];
  const [selectedStation, setSelectedStation] = useState<string>("");
  const [horizon, setHorizon] = useState<number>(60);

  const { data: detail, isLoading: loadingDetail } = useQuery({
    queryKey: ["station-detail", selectedStation],
    queryFn: () => fetchStationDetail(selectedStation),
    enabled: !!selectedStation,
    refetchInterval: 30000,
  });

  // Also fetch multi-horizon predictions for the selected station
  const { data: predData } = useQuery({
    queryKey: ["predictions", horizon],
    queryFn: () => fetchPredictions(horizon || undefined),
    refetchInterval: 30000,
  });

  if (loadingStations) return <div className="p-6 space-y-6"><CardSkeleton /><CardSkeleton /></div>;

  const forecast = detail?.forecast ?? [];
  const hourlyRidership = detail?.hourly_ridership ?? [];
  const filteredForecast = forecast.filter(f => new Date(f.timestamp).getTime() <= Date.now() + horizon * 60000);
  const maxVal = Math.max(...filteredForecast.map((f) => f.predicted), ...hourlyRidership.map((h) => h.ridership), 1);

  // Compute MAPE from forecast vs actual hourly ridership
  const { mape, mae, rmse, accuracyPct } = useMemo(() => {
    if (filteredForecast.length === 0 || hourlyRidership.length === 0) return { mape: null, mae: null, rmse: null, accuracyPct: null };
    const paired = filteredForecast
      .map((f) => {
        const actual = hourlyRidership.find((h) => h.hour === new Date(f.timestamp).getHours());
        return { predicted: f.predicted, actual: actual?.ridership };
      })
      .filter((p) => p.actual !== undefined && p.actual !== null);
    if (paired.length === 0) return { mape: null, mae: null, rmse: null, accuracyPct: null };
    const absErrors = paired.map((p) => Math.abs(p.predicted - (p.actual as number)));
    const pctErrors = paired.map((p) => Math.abs(p.predicted - (p.actual as number)) / Math.max(1, p.actual as number));
    const sumAbs = absErrors.reduce((a, b) => a + b, 0);
    const sumPct = pctErrors.reduce((a, b) => a + b, 0);
    const sumSq = absErrors.reduce((a, b) => a + b * b, 0);
    return {
      mape: (sumPct / paired.length) * 100,
      mae: sumAbs / paired.length,
      rmse: Math.sqrt(sumSq / paired.length),
      accuracyPct: Math.max(0, 100 - (sumPct / paired.length) * 100),
    };
  }, [filteredForecast, hourlyRidership]);

  // Chart data for forecast vs actual
  const comparisonData = filteredForecast.map((f) => {
    const actual = hourlyRidership.find((h) => h.hour === new Date(f.timestamp).getHours());
    return {
      hour: new Date(f.timestamp).getHours(),
      hourLabel: `${String(new Date(f.timestamp).getHours()).padStart(2, "0")}:00`,
      predicted: f.predicted,
      actual: actual?.ridership ?? null,
      confidence: f.confidence,
      upper: Math.round(f.predicted * (1 + (1 - f.confidence))),
      lower: Math.round(f.predicted * f.confidence),
    };
  });

  // Station-specific predictions
  const stationPreds = predData?.predictions?.filter((p) => p.station_id === selectedStation) ?? [];

  const accColor = (val: number | null) => {
    if (val === null) return "text-gray-400";
    if (val >= 90) return "text-green-600 dark:text-green-400";
    if (val >= 80) return "text-amber-600 dark:text-amber-400";
    return "text-red-600 dark:text-red-400";
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold dark:text-white">Station Forecast</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">Predict future passenger flow for any station with DTS-GSSF model validation.</p>
      </div>

      {/* Station + Horizon Selection */}
      <Card>
        <CardHeader><CardTitle className="text-sm">Select Station & Forecast Horizon</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1.5 dark:text-gray-300">Station</label>
              <select
                className="w-full border rounded-lg px-3 py-2.5 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 outline-none"
                value={selectedStation}
                onChange={(e) => setSelectedStation(e.target.value)}
              >
                <option value="">— Choose a station —</option>
                {stations.map((s) => (
                  <option key={s.id} value={s.id}>{s.name} ({s.district ?? "Unknown"})</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5 dark:text-gray-300">Forecast Horizon</label>
              <div className="flex flex-wrap gap-2">
                {HORIZONS.map((h) => (
                  <button
                    key={h.minutes}
                    onClick={() => setHorizon(h.minutes)}
                    className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-colors ${
                      horizon === h.minutes
                        ? "bg-blue-600 text-white shadow-sm"
                        : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
                    }`}
                  >
                    {h.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
          {!selectedStation && (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 text-sm text-blue-700 dark:text-blue-300">
              Select a station to view passenger flow predictions and model accuracy metrics.
            </div>
          )}
        </CardContent>
      </Card>

      {loadingDetail && selectedStation && <CardSkeleton />}

      {detail && selectedStation && (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <Users className="h-3 w-3" /> Station
                </div>
                <p className="text-lg font-bold truncate dark:text-white">{detail.station.name}</p>
                <p className="text-xs text-gray-400">{detail.station.district ?? "Unknown"} · {detail.station.ridership_24h?.toLocaleString() ?? "—"} pax/24h</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <Target className="h-3 w-3" /> Model Accuracy
                </div>
                <p className={`text-lg font-bold ${accColor(accuracyPct)}`}>
                  {accuracyPct !== null ? `${accuracyPct.toFixed(1)}%` : "—"}
                </p>
                <p className="text-xs text-gray-400">MAPE: {mape !== null ? `${mape.toFixed(1)}%` : "N/A"}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <BarChart3 className="h-3 w-3" /> Forecast Points
                </div>
                <p className="text-lg font-bold dark:text-white">{filteredForecast.length}</p>
                <p className="text-xs text-gray-400">{stationPreds.length} multi-horizon</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <TrendingUp className="h-3 w-3" /> Connected Routes
                </div>
                <p className="text-lg font-bold dark:text-white">{detail.connected_routes.length}</p>
                <div className="flex gap-1 mt-1 flex-wrap">
                  {detail.connected_routes.slice(0, 3).map((r) => (
                    <span key={r.id} className="w-2 h-2 rounded-full" style={{ backgroundColor: r.color ?? "#888" }} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Validation Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-green-600" /> Model Validation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">MAE</p>
                  <p className="text-xl font-bold dark:text-white">{mae !== null ? mae.toFixed(1) : "—"}</p>
                  <p className="text-[10px] text-gray-400">passengers</p>
                </div>
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">MAPE</p>
                  <p className={`text-xl font-bold ${mape !== null ? (mape < 10 ? "text-green-600 dark:text-green-400" : mape < 20 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400") : ""}`}>
                    {mape !== null ? `${mape.toFixed(1)}%` : "—"}
                  </p>
                  <p className="text-[10px] text-gray-400">error rate</p>
                </div>
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">RMSE</p>
                  <p className="text-xl font-bold dark:text-white">{rmse !== null ? rmse.toFixed(1) : "—"}</p>
                  <p className="text-[10px] text-gray-400">root sq. error</p>
                </div>
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Accuracy</p>
                  <p className={`text-xl font-bold ${accColor(accuracyPct)}`}>
                    {accuracyPct !== null ? `${accuracyPct.toFixed(1)}%` : "—"}
                  </p>
                  <p className="text-[10px] text-gray-400">overall</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Forecast vs Actual Chart with Confidence Band */}
          {comparisonData.length > 0 && (
            <Card>
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle>Predicted vs Actual Passenger Flow</CardTitle>
                <Badge variant="default" className="text-xs">+{HORIZONS.find((h) => h.minutes === horizon)?.label ?? `${horizon}m`}</Badge>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={320}>
                  <AreaChart data={comparisonData}>
                    <defs>
                      <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="hourLabel" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} domain={[0, maxVal]} />
                    <Tooltip
                      contentStyle={{ fontSize: 12, borderRadius: 8 }}
                      formatter={(value: unknown, name: string) => {
                        if (name === "upper" || name === "lower") return null;
                        return [typeof value === "number" ? `${value} pax` : "N/A", name === "predicted" ? "Forecast" : "Actual"];
                      }}
                      labelFormatter={(label: string) => `Time: ${label}`}
                    />
                    <Legend formatter={(value: string) => {
                      if (value === "predicted") return "Forecast";
                      if (value === "actual") return "Actual";
                      if (value === "upper") return "Upper Bound";
                      if (value === "lower") return "Lower Bound";
                      return value;
                    }} />
                    <Area type="monotone" dataKey="upper" stroke="none" fill="url(#confGrad)" />
                    <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" fillOpacity={0.8} />
                    <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2.5} dot={false} />
                    <Line type="monotone" dataKey="actual" stroke="#34d399" strokeWidth={2} dot={false} connectNulls strokeDasharray="5 5" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Detailed Forecast Table */}
          {filteredForecast.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Detailed Forecast Table</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b dark:border-gray-700 text-left">
                        <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400">Time</th>
                        <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400 text-right">Predicted</th>
                        <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400 text-right">Confidence</th>
                        <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400 text-right">Range</th>
                        <th className="pb-2 font-semibold text-gray-500 dark:text-gray-400 text-right">Actual</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredForecast.slice(0, 24).map((f, i) => {
                        const actual = hourlyRidership.find((h) => h.hour === new Date(f.timestamp).getHours());
                        const upper = Math.round(f.predicted * (1 + (1 - f.confidence)));
                        const lower = Math.round(f.predicted * f.confidence);
                        const errPct = actual ? Math.abs(f.predicted - actual.ridership) / Math.max(1, actual.ridership) * 100 : null;
                        return (
                          <tr key={i} className="border-b dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                            <td className="py-2 pr-4 font-medium dark:text-white">
                              {new Date(f.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                            </td>
                            <td className="py-2 pr-4 text-right font-mono dark:text-gray-300">{f.predicted} pax</td>
                            <td className="py-2 pr-4 text-right">
                              <div className="flex items-center justify-end gap-1.5">
                                <div className="w-14 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${f.confidence * 100}%` }} />
                                </div>
                                <span className="text-xs font-mono text-gray-500">{(f.confidence * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="py-2 pr-4 text-right font-mono text-xs text-gray-400">{lower}–{upper}</td>
                            <td className="py-2 text-right font-mono dark:text-gray-300">
                              {actual ? actual.ridership : "—"}
                              {errPct !== null && (
                                <span className={`ml-1 text-[10px] ${errPct < 10 ? "text-green-500" : errPct < 20 ? "text-amber-500" : "text-red-500"}`}>
                                  ({errPct.toFixed(0)}%)
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Active Alerts */}
          {detail.alerts.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-red-500" /> Active Alerts for This Station
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {detail.alerts.map((a, i) => (
                  <div key={i} className={`p-3 rounded-lg text-sm ${
                    a.severity === "critical" ? "bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300" :
                    a.severity === "warning" ? "bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300" :
                    "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                  }`}>
                    <span className="font-semibold uppercase text-xs">{a.severity}</span>
                    <span className="ml-2">{a.title}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}