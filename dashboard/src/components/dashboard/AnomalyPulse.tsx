import { useState } from "react";
import { useRichAlerts } from "@/hooks/useRichAlerts";
import { useSimulationStore } from "@/stores/simulationStore";
import type { RichAlert, DriftAlert } from "@/types";

const severityConfig: Record<string, { bg: string; border: string; icon: string; pulse: string }> = {
  critical: { bg: "bg-destructive/10", border: "border-l-destructive", icon: "🔴", pulse: "animate-pulse" },
  warning: { bg: "bg-chart-4/10", border: "border-l-chart-4", icon: "🟡", pulse: "" },
  info: { bg: "bg-primary/10", border: "border-l-primary", icon: "🔵", pulse: "" },
};

const familyIcons: Record<string, string> = {
  station: "📍",
  route: "🚌",
  forecast: "📈",
  system: "⚙️",
};

function confidenceBar(confidence?: number) {
  if (confidence == null) return null;
  const pct = Math.round(confidence * 100);
  const color = pct >= 85 ? "bg-chart-2" : pct >= 65 ? "bg-chart-4" : "bg-destructive";
  return (
    <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
      <div className={`h-full rounded-full ${color} transition-all duration-500`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function formatSla(minutes?: number): string {
  if (minutes == null) return "";
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export default function AnomalyPulse() {
  const { criticalAlerts, unacknowledgedAlerts } = useRichAlerts();
  const driftAlerts = useSimulationStore((s) => s.driftAlerts);
  const [expanded, setExpanded] = useState(false);

  const anomalyCount = unacknowledgedAlerts.length + driftAlerts.length;
  const hasCritical = criticalAlerts.length > 0 || driftAlerts.some((d) => d.severity === "high");

  type AlertItem = (RichAlert & { _type: "alert" }) | (DriftAlert & { _type: "drift" });
  const allItems: AlertItem[] = [
    ...unacknowledgedAlerts.map((a) => ({ ...a, _type: "alert" as const })),
    ...driftAlerts.map((d) => ({ ...d, _type: "drift" as const })),
  ];

  return (
    <div className="space-y-2">
      {/* Pulse indicator */}
      <button
        onClick={() => setExpanded(!expanded)}
        className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold transition-all w-full text-left ${
          hasCritical
            ? "bg-destructive/10 border border-destructive/30"
            : anomalyCount > 0
              ? "bg-chart-4/10 border border-chart-4/30"
              : "bg-muted border border-border"
        }`}
      >
        <span className={`w-2.5 h-2.5 rounded-full ${hasCritical ? "bg-destructive animate-pulse" : anomalyCount > 0 ? "bg-chart-4" : "bg-chart-2"}`} />
        <span className="flex-1">{anomalyCount > 0 ? `${anomalyCount} anomal${anomalyCount === 1 ? "y" : "ies"}` : "No anomalies"}</span>
        <span className="text-muted-foreground text-xs">{expanded ? "▲" : "▼"}</span>
      </button>

      {/* Expanded feed */}
      {expanded && allItems.length > 0 && (
        <div className="space-y-1.5 max-h-64 overflow-y-auto">
          {allItems.slice(0, 20).map((item, i) => {
            if ("_type" in item && item._type === "drift") {
              const d = item as DriftAlert & { _type: "drift" };
              return (
                <div key={`drift-${i}`} className="p-2 rounded-lg border-l-4 border-l-chart-5 bg-chart-5/5 text-xs">
                  <div className="flex items-center gap-1.5 font-semibold">
                    <span>📊</span>
                    <span>Drift: {d.metric}</span>
                    <span className="ml-auto text-muted-foreground">{Math.round(d.deviation_pct)}% deviation</span>
                  </div>
                  <div className="text-muted-foreground mt-0.5">
                    Value: {d.current_value.toFixed(2)} vs baseline {d.baseline_value.toFixed(2)}
                  </div>
                </div>
              );
            }

            const alert = item as RichAlert & { _type: "alert" };
            const config = severityConfig[alert.severity] ?? severityConfig.info;
            const icon = familyIcons[alert.family ?? "system"] ?? "⚠️";

            return (
              <div key={`alert-${alert.id}`} className={`p-2.5 rounded-lg border-l-4 ${config.border} ${config.bg} text-xs`}>
                <div className="flex items-center gap-1.5 font-semibold">
                  <span>{icon}</span>
                  <span className="flex-1 truncate">{alert.title}</span>
                  {alert.confidence != null && (
                    <span className="text-[10px] text-muted-foreground font-mono">{Math.round(alert.confidence * 100)}%</span>
                  )}
                </div>
                {alert.why && <p className="text-muted-foreground mt-0.5">{alert.why}</p>}
                {alert.confidence != null && confidenceBar(alert.confidence)}
                {alert.consequence_if_ignored && (
                  <p className="text-destructive mt-0.5 font-medium">⚠ {alert.consequence_if_ignored}</p>
                )}
                {alert.recommended_actions && alert.recommended_actions.length > 0 && (
                  <div className="flex gap-1 mt-1.5 flex-wrap">
                    {alert.recommended_actions.map((action, j) => (
                      <span key={j} className="px-2 py-0.5 rounded-full bg-chart-2/20 text-foreground text-[10px] font-medium">
                        {action.label}
                      </span>
                    ))}
                  </div>
                )}
                {alert.sla_timer_minutes && (
                  <div className="mt-1 text-muted-foreground font-mono text-[10px]">
                    ⏱ SLA: {formatSla(alert.sla_timer_minutes)}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}