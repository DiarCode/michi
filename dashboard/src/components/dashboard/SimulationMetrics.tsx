import { useSimulationStore } from "@/stores/simulationStore";
import { useConnectionStore } from "@/stores/connectionStore";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { TrendingUp, TrendingDown, Minus, Activity, AlertTriangle } from "lucide-react";

/** Colour for drift status badge */
function driftColor(status: string) {
  if (status === "critical") return "bg-red-500 text-white";
  if (status === "warning") return "bg-amber-500 text-white";
  return "bg-green-500 text-white";
}

/** Trend arrow for a metric vs. previous value */
function TrendArrow({ current, previous }: { current: number; previous: number | undefined }) {
  if (previous === undefined) return <Minus className="h-3 w-3 text-gray-400" />;
  if (current < previous) return <TrendingDown className="h-3 w-3 text-green-500" />;
  if (current > previous) return <TrendingUp className="h-3 w-3 text-red-500" />;
  return <Minus className="h-3 w-3 text-gray-400" />;
}

/** Format timestamp for chart x-axis */
function fmtTime(ts: string | undefined) {
  if (!ts) return "";
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

export default function SimulationMetrics() {
  const { tick, metricsHistory, driftAlerts, isStale, lastTickAt } =
    useSimulationStore();
  const connected = useConnectionStore((s) => s.connected);

  const latest = metricsHistory[metricsHistory.length - 1];
  const previous = metricsHistory.length >= 2 ? metricsHistory[metricsHistory.length - 2] : undefined;

  // Keep only last 5 minutes (at ~1 tick/sec that is 300 points, but we cap at 100 for perf)
  const chartData = metricsHistory.slice(-100).map((m, i) => ({
    idx: i,
    time: fmtTime(m.timestamp as string | undefined),
    mae: m.mae,
    mape: m.mape,
  }));

  // Derive drift status from latest metrics
  const driftStatus = latest?.mape !== undefined
    ? latest.mape > 15 ? "critical" : latest.mape > 10 ? "warning" : "normal"
    : "normal";

  return (
    <div className="space-y-4">
      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">MAE</p>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold dark:text-white">
                {latest?.mae !== undefined ? latest.mae.toFixed(2) : "--"}
              </span>
              <TrendArrow current={latest?.mae ?? 0} previous={previous?.mae} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">MAPE</p>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold dark:text-white">
                {latest?.mape !== undefined ? latest.mape.toFixed(2) : "--"}
              </span>
              <TrendArrow current={latest?.mape ?? 0} previous={previous?.mape} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Accuracy</p>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold dark:text-white">
                {latest?.accuracy !== undefined ? latest.accuracy.toFixed(2) : "--"}%
              </span>
              <TrendArrow
                current={-(latest?.accuracy ?? 0)}
                previous={previous?.accuracy !== undefined ? -previous.accuracy : undefined}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs text-gray-500 dark:text-gray-400">Drift Status</p>
              <Activity className="h-3.5 w-3.5 text-gray-400" />
            </div>
            <div className="flex items-center gap-2">
              <Badge className={driftColor(driftStatus)}>
                {driftStatus.toUpperCase()}
              </Badge>
              {driftAlerts.length > 0 && (
                <span className="text-xs text-amber-600 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  {driftAlerts.length}
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Connection & tick info */}
      <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
        <span className="flex items-center gap-1">
          <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
          {connected ? "Connected" : "Disconnected"}
        </span>
        <span>Tick #{tick}</span>
        {lastTickAt && <span>Last: {new Date(lastTickAt).toLocaleTimeString()}</span>}
        {isStale && <span className="text-amber-500 font-medium">Stale</span>}
      </div>

      {/* MAE / MAPE time series chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">MAE / MAPE Time Series</CardTitle>
        </CardHeader>
        <CardContent>
          {chartData.length < 2 ? (
            <p className="text-sm text-gray-400 text-center py-8">
              Waiting for simulation data...
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                  labelFormatter={(l: string) => `Time: ${l}`}
                />
                <Line type="monotone" dataKey="mae" stroke="#3b82f6" strokeWidth={2} dot={false} name="MAE" />
                <Line type="monotone" dataKey="mape" stroke="#f59e0b" strokeWidth={2} dot={false} name="MAPE" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>
    </div>
  );
}