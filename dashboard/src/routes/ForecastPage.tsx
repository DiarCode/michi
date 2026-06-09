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

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-michi-dark text-white rounded-xl px-3.5 py-2.5 shadow-tooltip text-xs">
      <p className="text-michi-muted mb-1.5">{label}</p>
      {payload.filter((p: any) => p.dataKey !== "upper" && p.dataKey !== "lower").map((entry: any, i: number) => (
        <p key={i} className="font-semibold">
          {entry.name === "predicted" ? "Forecast" : entry.name === "actual" ? "Actual" : entry.name}: {typeof entry.value === "number" ? `${entry.value} pax` : "N/A"}
        </p>
      ))}
    </div>
  );
};

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

  const { data: predData } = useQuery({
    queryKey: ["predictions", horizon],
    queryFn: () => fetchPredictions(horizon || undefined),
    refetchInterval: 30000,
  });

  if (loadingStations) return <div className="p-8 space-y-8"><CardSkeleton /><CardSkeleton /></div>;

  const forecast = detail?.forecast ?? [];
  const hourlyRidership = detail?.hourly_ridership ?? [];
  const filteredForecast = forecast.filter(f => new Date(f.timestamp).getTime() <= Date.now() + horizon * 60000);
  const maxVal = Math.max(...filteredForecast.map((f) => f.predicted), ...hourlyRidership.map((h) => h.ridership), 1);

  const { mape, mae, rmse, accuracyPct } = useMemo(() => {
    if (filteredForecast.length === 0 || hourlyRidership.length === 0) return { mape: null, mae: null, rmse: null, accuracyPct: null };
    const paired = filteredForecast
      .map((f) => { const actual = hourlyRidership.find((h) => h.hour === new Date(f.timestamp).getHours()); return { predicted: f.predicted, actual: actual?.ridership }; })
      .filter((p) => p.actual !== undefined && p.actual !== null);
    if (paired.length === 0) return { mape: null, mae: null, rmse: null, accuracyPct: null };
    const absErrors = paired.map((p) => Math.abs(p.predicted - (p.actual as number)));
    const pctErrors = paired.map((p) => Math.abs(p.predicted - (p.actual as number)) / Math.max(1, p.actual as number));
    const sumAbs = absErrors.reduce((a, b) => a + b, 0);
    const sumPct = pctErrors.reduce((a, b) => a + b, 0);
    const sumSq = absErrors.reduce((a, b) => a + b * b, 0);
    return { mape: (sumPct / paired.length) * 100, mae: sumAbs / paired.length, rmse: Math.sqrt(sumSq / paired.length), accuracyPct: Math.max(0, 100 - (sumPct / paired.length) * 100) };
  }, [filteredForecast, hourlyRidership]);

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

  const stationPreds = predData?.predictions?.filter((p) => p.station_id === selectedStation) ?? [];

  const accColor = (val: number | null) => {
    if (val === null) return "text-michi-muted";
    if (val >= 90) return "text-michi-lime-dark";
    if (val >= 80) return "text-michi-amber";
    return "text-michi-red";
  };

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Station Forecast</h1>
        <p className="text-base text-michi-muted mt-1">Predict future passenger flow for any station with DTS-GSSF model validation</p>
      </div>

      <Card>
        <CardHeader><CardTitle>Select Station & Forecast Horizon</CardTitle></CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div>
              <label className="block text-sm font-semibold text-michi-dark mb-2">Station</label>
              <select
                className="w-full border border-michi-border rounded-xl px-4 py-3 bg-white text-michi-dark font-medium focus:ring-2 focus:ring-michi-lime/50 focus:border-michi-lime outline-none"
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
              <label className="block text-sm font-semibold text-michi-dark mb-2">Forecast Horizon</label>
              <div className="flex flex-wrap gap-2">
                {HORIZONS.map((h) => (
                  <button
                    key={h.minutes}
                    onClick={() => setHorizon(h.minutes)}
                    className={`px-4 py-2 text-sm rounded-full font-semibold transition-all ${
                      horizon === h.minutes
                        ? "bg-michi-dark text-white shadow-sm"
                        : "bg-michi-warm text-michi-body border border-michi-border hover:bg-michi-border"
                    }`}
                  >
                    {h.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
          {!selectedStation && (
            <div className="bg-michi-lime/10 border border-michi-lime/30 rounded-xl p-4 text-sm text-michi-lime-dark font-medium">
              Select a station to view passenger flow predictions and model accuracy metrics.
            </div>
          )}
        </CardContent>
      </Card>

      {loadingDetail && selectedStation && <CardSkeleton />}

      {detail && selectedStation && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <Users size={14} /> Station
                </div>
                <p className="text-xl font-extrabold text-michi-dark truncate">{detail.station.name}</p>
                <p className="text-sm text-michi-muted mt-0.5">{detail.station.district ?? "Unknown"} · {detail.station.ridership_24h?.toLocaleString() ?? "—"} pax/24h</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <Target size={14} /> Model Accuracy
                </div>
                <p className={`text-xl font-extrabold ${accColor(accuracyPct)}`}>
                  {accuracyPct !== null ? `${accuracyPct.toFixed(1)}%` : "—"}
                </p>
                <p className="text-sm text-michi-muted mt-0.5">MAPE: {mape !== null ? `${mape.toFixed(1)}%` : "N/A"}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <BarChart3 size={14} /> Forecast Points
                </div>
                <p className="text-xl font-extrabold text-michi-dark">{filteredForecast.length}</p>
                <p className="text-sm text-michi-muted mt-0.5">{stationPreds.length} multi-horizon</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <TrendingUp size={14} /> Connected Routes
                </div>
                <p className="text-xl font-extrabold text-michi-dark">{detail.connected_routes.length}</p>
                <div className="flex gap-1.5 mt-1.5 flex-wrap">
                  {detail.connected_routes.slice(0, 3).map((r) => (
                    <span key={r.id} className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: r.color ?? "#888" }} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle2 size={18} className="text-michi-lime-dark" /> Model Validation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-michi-warm rounded-xl">
                  <p className="text-sm text-michi-muted font-medium mb-1">MAE</p>
                  <p className="text-2xl font-extrabold text-michi-dark">{mae !== null ? mae.toFixed(1) : "—"}</p>
                  <p className="text-xs text-michi-muted mt-0.5">passengers</p>
                </div>
                <div className="text-center p-4 bg-michi-warm rounded-xl">
                  <p className="text-sm text-michi-muted font-medium mb-1">MAPE</p>
                  <p className={`text-2xl font-extrabold ${mape !== null ? (mape < 10 ? "text-michi-lime-dark" : mape < 20 ? "text-michi-amber" : "text-michi-red") : "text-michi-dark"}`}>
                    {mape !== null ? `${mape.toFixed(1)}%` : "—"}
                  </p>
                  <p className="text-xs text-michi-muted mt-0.5">error rate</p>
                </div>
                <div className="text-center p-4 bg-michi-warm rounded-xl">
                  <p className="text-sm text-michi-muted font-medium mb-1">RMSE</p>
                  <p className="text-2xl font-extrabold text-michi-dark">{rmse !== null ? rmse.toFixed(1) : "—"}</p>
                  <p className="text-xs text-michi-muted mt-0.5">root sq. error</p>
                </div>
                <div className="text-center p-4 bg-michi-warm rounded-xl">
                  <p className="text-sm text-michi-muted font-medium mb-1">Accuracy</p>
                  <p className={`text-2xl font-extrabold ${accColor(accuracyPct)}`}>
                    {accuracyPct !== null ? `${accuracyPct.toFixed(1)}%` : "—"}
                  </p>
                  <p className="text-xs text-michi-muted mt-0.5">overall</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {comparisonData.length > 0 && (
            <Card>
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle>Predicted vs Actual Passenger Flow</CardTitle>
                <Badge variant="success">+{HORIZONS.find((h) => h.minutes === horizon)?.label ?? `${horizon}m`}</Badge>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={340}>
                  <AreaChart data={comparisonData}>
                    <defs>
                      <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#B1E743" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#B1E743" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E8E8E0" />
                    <XAxis dataKey="hourLabel" tick={{ fontSize: 12, fill: '#9C9C95' }} />
                    <YAxis tick={{ fontSize: 12, fill: '#9C9C95' }} domain={[0, maxVal]} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend formatter={(value: string) => {
                      if (value === "predicted") return "Forecast";
                      if (value === "actual") return "Actual";
                      if (value === "upper") return "Upper Bound";
                      if (value === "lower") return "Lower Bound";
                      return value;
                    }} />
                    <Area type="monotone" dataKey="upper" stroke="none" fill="url(#confGrad)" />
                    <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" fillOpacity={0.8} />
                    <Line type="monotone" dataKey="predicted" stroke="#B1E743" strokeWidth={2.5} dot={false} />
                    <Line type="monotone" dataKey="actual" stroke="#9C9C95" strokeWidth={2} dot={false} connectNulls strokeDasharray="5 5" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {filteredForecast.length > 0 && (
            <Card>
              <CardHeader><CardTitle>Detailed Forecast Table</CardTitle></CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-michi-border text-left">
                        <th className="pb-3 pr-4 font-semibold text-michi-muted">Time</th>
                        <th className="pb-3 pr-4 font-semibold text-michi-muted text-right">Predicted</th>
                        <th className="pb-3 pr-4 font-semibold text-michi-muted text-right">Confidence</th>
                        <th className="pb-3 pr-4 font-semibold text-michi-muted text-right">Range</th>
                        <th className="pb-3 font-semibold text-michi-muted text-right">Actual</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredForecast.slice(0, 24).map((f, i) => {
                        const actual = hourlyRidership.find((h) => h.hour === new Date(f.timestamp).getHours());
                        const upper = Math.round(f.predicted * (1 + (1 - f.confidence)));
                        const lower = Math.round(f.predicted * f.confidence);
                        const errPct = actual ? Math.abs(f.predicted - actual.ridership) / Math.max(1, actual.ridership) * 100 : null;
                        return (
                          <tr key={i} className="border-b border-michi-border/50 hover:bg-michi-warm transition-colors">
                            <td className="py-3 pr-4 font-semibold text-michi-dark">
                              {new Date(f.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                            </td>
                            <td className="py-3 pr-4 text-right font-mono text-michi-body">{f.predicted} pax</td>
                            <td className="py-3 pr-4 text-right">
                              <div className="flex items-center justify-end gap-2">
                                <div className="w-16 h-2.5 bg-michi-warm rounded-full overflow-hidden">
                                  <div className="h-full bg-michi-lime rounded-full" style={{ width: `${f.confidence * 100}%` }} />
                                </div>
                                <span className="text-xs font-mono text-michi-muted">{(f.confidence * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="py-3 pr-4 text-right font-mono text-xs text-michi-muted">{lower}–{upper}</td>
                            <td className="py-3 text-right font-mono text-michi-body">
                              {actual ? actual.ridership : "—"}
                              {errPct !== null && (
                                <span className={`ml-1.5 text-xs font-semibold ${errPct < 10 ? "text-michi-lime-dark" : errPct < 20 ? "text-michi-amber" : "text-michi-red"}`}>
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

          {detail.alerts.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle size={18} className="text-michi-red" /> Active Alerts for This Station
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2.5">
                {detail.alerts.map((a, i) => (
                  <div key={i} className={`p-3.5 rounded-xl text-sm border-l-4 ${
                    a.severity === "critical" ? "bg-michi-red/8 border-l-michi-red text-michi-red" :
                    a.severity === "warning" ? "bg-michi-amber/8 border-l-michi-amber text-michi-amber" :
                    "bg-michi-warm border-l-michi-muted text-michi-body"
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