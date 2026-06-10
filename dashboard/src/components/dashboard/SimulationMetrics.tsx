import { useSimulationStore } from "@/stores/simulationStore";
import { useConnectionStore } from "@/stores/connectionStore";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowUp01Icon, ArrowDown01Icon, MinusSignIcon, ActivityIcon, Alert01Icon } from "@/lib/icons";

function driftColor(status: string) {
  if (status === "critical") return "bg-destructive text-white";
  if (status === "warning") return "bg-chart-4 text-white";
  return "bg-chart-2 text-foreground";
}

function TrendArrow({ current, previous }: { current: number; previous: number | undefined }) {
  if (previous === undefined) return <HugeiconsIcon icon={MinusSignIcon} className="h-3 w-3 text-muted-foreground" />;
  if (current < previous) return <HugeiconsIcon icon={ArrowDown01Icon} className="h-3 w-3 text-chart-2" />;
  if (current > previous) return <HugeiconsIcon icon={ArrowUp01Icon} className="h-3 w-3 text-destructive" />;
  return <HugeiconsIcon icon={MinusSignIcon} className="h-3 w-3 text-muted-foreground" />;
}

function fmtTime(ts: string | undefined) {
  if (!ts) return "";
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-foreground text-background rounded-xl px-3 py-2 shadow-lg text-xs">
      <p className="text-muted-foreground mb-1">{label}</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} className="font-semibold">
          {entry.name}: {entry.value.toFixed(2)}
        </p>
      ))}
    </div>
  );
};

export default function SimulationMetrics() {
  const { tick, metricsHistory, driftAlerts, isStale, lastTickAt } =
    useSimulationStore();
  const connected = useConnectionStore((s) => s.connected);

  const latest = metricsHistory[metricsHistory.length - 1];
  const previous = metricsHistory.length >= 2 ? metricsHistory[metricsHistory.length - 2] : undefined;

  const chartData = metricsHistory.slice(-100).map((m, i) => ({
    idx: i,
    time: fmtTime(m.timestamp as string | undefined),
    mae: m.mae,
    mape: m.mape,
  }));

  const driftStatus = latest?.mape !== undefined
    ? latest.mape > 15 ? "critical" : latest.mape > 10 ? "warning" : "normal"
    : "normal";

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-muted-foreground font-medium">MAE</span>
            <div className="flex items-center gap-2 mt-2">
              <span className="text-3xl font-extrabold text-foreground">
                {latest?.mae !== undefined ? latest.mae.toFixed(2) : "—"}
              </span>
              <TrendArrow current={latest?.mae ?? 0} previous={previous?.mae} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-muted-foreground font-medium">MAPE</span>
            <div className="flex items-center gap-2 mt-2">
              <span className="text-3xl font-extrabold text-foreground">
                {latest?.mape !== undefined ? latest.mape.toFixed(2) : "—"}
              </span>
              <TrendArrow current={latest?.mape ?? 0} previous={previous?.mape} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-muted-foreground font-medium">Accuracy</span>
            <div className="flex items-center gap-2 mt-2">
              <span className="text-3xl font-extrabold text-foreground">
                {latest?.accuracy !== undefined ? latest.accuracy.toFixed(2) : "—"}%
              </span>
              <TrendArrow
                current={-(latest?.accuracy ?? 0)}
                previous={previous?.accuracy !== undefined ? -previous.accuracy : undefined}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground font-medium">Drift Status</span>
              <HugeiconsIcon icon={ActivityIcon} size={16} className="text-muted-foreground" />
            </div>
            <div className="flex items-center gap-2 mt-2">
              <Badge className={driftColor(driftStatus)}>
                {driftStatus.toUpperCase()}
              </Badge>
              {driftAlerts.length > 0 && (
                <span className="text-sm text-chart-4 font-semibold flex items-center gap-1">
                  <HugeiconsIcon icon={Alert01Icon} size={14} />
                  {driftAlerts.length}
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="flex items-center gap-5 text-sm text-muted-foreground font-medium">
        <span className="flex items-center gap-1.5">
          <span className={`w-2.5 h-2.5 rounded-full ${connected ? "bg-chart-2" : "bg-destructive"}`} />
          {connected ? "Connected" : "Disconnected"}
        </span>
        <span>Tick #{tick}</span>
        {lastTickAt && <span>Last: {new Date(lastTickAt).toLocaleTimeString()}</span>}
        {isStale && <span className="text-chart-4 font-semibold">Stale</span>}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>MAE / MAPE Time Series</CardTitle>
        </CardHeader>
        <CardContent>
          {chartData.length < 2 ? (
            <p className="text-base text-muted-foreground text-center py-10">
              Waiting for simulation data...
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="time" tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} interval="preserveStartEnd" stroke="var(--border)" />
                <YAxis tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} stroke="var(--border)" />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="mae" stroke="var(--chart-2)" strokeWidth={2.5} dot={false} name="MAE" />
                <Line type="monotone" dataKey="mape" stroke="var(--chart-4)" strokeWidth={2.5} dot={false} name="MAPE" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>
    </div>
  );
}