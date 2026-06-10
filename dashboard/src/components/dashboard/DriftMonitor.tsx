import { useSimulationStore } from "@/stores/simulationStore";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function DriftMonitor() {
  const metricsHistory = useSimulationStore((s) => s.metricsHistory);
  const running = useSimulationStore((s) => s.running);

  if (metricsHistory.length === 0 && !running) return null;

  // Get last 20 MAPE values for sparkline
  const recentMetrics = metricsHistory.slice(-20);
  const lastMetric = recentMetrics[recentMetrics.length - 1];
  const driftStatus = lastMetric?.drift_status ?? "normal";

  // Compute status color
  const statusColor =
    driftStatus === "critical" ? "text-destructive" :
    driftStatus === "warning" ? "text-chart-4" :
    "text-chart-2";

  const statusLabel =
    driftStatus === "critical" ? "CRITICAL DRIFT" :
    driftStatus === "warning" ? "DRIFT WARNING" :
    "NORMAL";

  const borderColor =
    driftStatus === "critical" ? "border-destructive" :
    driftStatus === "warning" ? "border-chart-4" :
    "border-chart-2";

  const strokeColor =
    driftStatus === "critical" ? "var(--destructive)" :
    driftStatus === "warning" ? "var(--chart-4)" :
    "var(--chart-2)";

  const warnStroke = "var(--chart-4)";
  const critStroke = "var(--destructive)";

  // Build sparkline path
  const mapeValues = recentMetrics.map((m) => m.mape);
  const maxMape = Math.max(...mapeValues, 10);
  const sparkH = 24;
  const sparkW = 100;

  const points = mapeValues.map((v, i) => {
    const x = (i / (mapeValues.length - 1 || 1)) * sparkW;
    const y = sparkH - (v / maxMape) * sparkH;
    return `${x},${y}`;
  }).join(" ");

  return (
    <Card className={`border-2 ${borderColor} ${driftStatus === "critical" ? "animate-pulse" : ""}`}>
      <CardHeader className="pb-1">
        <CardTitle className="flex items-center justify-between text-sm">
          <span>Model Drift</span>
          <span className={`text-xs font-mono font-bold ${statusColor}`}>
            {statusLabel}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        {mapeValues.length > 1 && (
          <svg viewBox={`0 0 ${sparkW} ${sparkH + 4}`} className="w-full h-8 mt-1">
            {/* Warning threshold line at 10% */}
            <line
              x1="0" y1={sparkH - (10 / maxMape) * sparkH + 2}
              x2={sparkW} y2={sparkH - (10 / maxMape) * sparkH + 2}
              stroke={warnStroke} strokeWidth="0.5" strokeDasharray="2,2"
            />
            {/* Critical threshold line at 15% */}
            <line
              x1="0" y1={sparkH - (15 / maxMape) * sparkH + 2}
              x2={sparkW} y2={sparkH - (15 / maxMape) * sparkH + 2}
              stroke={critStroke} strokeWidth="0.5" strokeDasharray="2,2"
            />
            {/* MAPE sparkline */}
            <polyline
              points={points}
              fill="none"
              stroke={strokeColor}
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              transform={`translate(0, 2)`}
            />
          </svg>
        )}
        <div className="flex items-center justify-between text-xs text-muted-foreground mt-1">
          <span>MAPE: {lastMetric ? `${lastMetric.mape.toFixed(1)}%` : "—"}</span>
          <span>Accuracy: {lastMetric?.accuracy != null ? `${lastMetric.accuracy.toFixed(1)}%` : "—"}</span>
        </div>
      </CardContent>
    </Card>
  );
}