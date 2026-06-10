import { useQuery } from "@tanstack/react-query";
import KPIGrid from "@/components/dashboard/KPIGrid";
import PredictiveHeatmap from "@/components/dashboard/PredictiveHeatmap";
import AnomalyPulse from "@/components/dashboard/AnomalyPulse";
import DriftMonitor from "@/components/dashboard/DriftMonitor";
import ConnectionProtectionPanel from "@/components/dashboard/ConnectionProtectionPanel";
import PlaybookCard from "@/components/dashboard/PlaybookCard";
import InterventionTracker from "@/components/dashboard/InterventionTracker";
import { fetchSuggestions } from "@/lib/api";
import { useRichAlerts } from "@/hooks/useRichAlerts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { HugeiconsIcon } from "@hugeicons/react";
import { LightbulbOffIcon, ActivityIcon } from "@/lib/icons";
import { Link } from "react-router-dom";
import { useSimulationStore } from "@/stores/simulationStore";
import type { Suggestion } from "@/types";

function priorityColor(priority: string) {
  if (priority === "critical" || priority === "high") return "border-l-destructive";
  if (priority === "medium") return "border-l-chart-4";
  return "border-l-chart-1";
}

function MiniSimulationCard() {
  const { running, tick, metricsHistory } = useSimulationStore();
  const latest = metricsHistory[metricsHistory.length - 1];

  const driftStatus = latest?.mape !== undefined
    ? latest.mape > 15 ? "critical" : latest.mape > 10 ? "warning" : "normal"
    : "normal";

  const driftBadge = driftStatus === "critical"
    ? "bg-destructive text-white"
    : driftStatus === "warning"
    ? "bg-chart-4 text-white"
    : "bg-chart-2 text-foreground";

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between pb-2">
        <CardTitle className="flex items-center gap-2">
          <HugeiconsIcon icon={ActivityIcon} size={18} className="text-chart-2" />
          Simulation Engine
        </CardTitle>
        <Link to="/simulation" className="text-sm text-chart-2 hover:underline font-semibold">
          View details →
        </Link>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-sm text-muted-foreground mb-1">Status</p>
            <Badge className={running ? "bg-chart-2 text-foreground" : "bg-border text-muted-foreground"}>
              {running ? "Running" : "Stopped"}
            </Badge>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">Tick</p>
            <p className="text-2xl font-extrabold text-foreground">{tick}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">MAPE</p>
            <p className="text-2xl font-extrabold text-foreground">
              {latest?.mape !== undefined ? latest.mape.toFixed(1) : "—"}
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">Drift</p>
            <Badge className={driftBadge}>{driftStatus.toUpperCase()}</Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function CommandCenter() {
  const { data: suggestionsData } = useQuery({
    queryKey: ["suggestions"],
    queryFn: fetchSuggestions,
    refetchInterval: 30000,
  });
  const { criticalAlerts } = useRichAlerts();

  const suggestions = suggestionsData?.suggestions ?? [];

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-foreground">Command Center</h1>
      </div>

      <KPIGrid />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PredictiveHeatmap />
        <div className="space-y-4">
          <AnomalyPulse />
          <DriftMonitor />
          <ConnectionProtectionPanel />
        </div>
      </div>

      {/* Playbook cards for critical alerts */}
      {criticalAlerts.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-bold text-foreground flex items-center gap-2">
            🔴 Active Playbooks
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {criticalAlerts.slice(0, 4).map((alert) => (
              <PlaybookCard
                key={alert.id}
                alert={alert}
                routeId={alert.route_id}
                stationId={alert.station_id}
              />
            ))}
          </div>
        </div>
      )}

      <InterventionTracker />

      <MiniSimulationCard />

      {suggestions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <HugeiconsIcon icon={LightbulbOffIcon} size={18} className="text-chart-4" /> Optimization Suggestions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-72 overflow-y-auto">
              {suggestions.slice(0, 8).map((s: Suggestion, i: number) => (
                <div key={i} className={`p-4 rounded-xl border-l-4 ${priorityColor(s.priority)} bg-muted`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant={s.priority === "high" ? "destructive" : s.priority === "medium" ? "secondary" : "default"}>
                        {s.priority}
                      </Badge>
                      <span className="text-sm font-semibold text-foreground">{s.title}</span>
                    </div>
                    <span className="text-xs text-muted-foreground font-medium">{s.type}</span>
                  </div>
                  <p className="text-sm text-muted-foreground mt-1.5">{s.description}</p>
                  {s.action && <p className="text-sm text-chart-2 font-semibold mt-1">{s.action}</p>}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}