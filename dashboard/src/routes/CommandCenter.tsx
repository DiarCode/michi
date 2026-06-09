import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import KPIGrid from "@/components/dashboard/KPIGrid";
import CongestionHeatmap from "@/components/dashboard/CongestionHeatmap";
import AlertTicker from "@/components/dashboard/AlertTicker";
import { fetchSuggestions, fetchInterventions, updateInterventionStatus } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Lightbulb, Zap, CheckCircle, Clock, XCircle, Activity } from "lucide-react";
import { Link } from "react-router-dom";
import { useSimulationStore } from "@/stores/simulationStore";
import type { Suggestion, Intervention } from "@/types";

function statusIcon(status: string) {
  switch (status) {
    case "completed": return <CheckCircle className="h-4 w-4 text-michi-lime-dark" />;
    case "approved": return <CheckCircle className="h-4 w-4 text-michi-teal" />;
    case "executing": return <Zap className="h-4 w-4 text-michi-amber" />;
    case "cancelled": return <XCircle className="h-4 w-4 text-michi-muted" />;
    default: return <Clock className="h-4 w-4 text-michi-muted" />;
  }
}

function priorityColor(priority: string) {
  if (priority === "critical" || priority === "high") return "border-l-michi-red";
  if (priority === "medium") return "border-l-michi-amber";
  return "border-l-michi-teal";
}

function MiniSimulationCard() {
  const { running, tick, metricsHistory } = useSimulationStore();
  const latest = metricsHistory[metricsHistory.length - 1];

  const driftStatus = latest?.mape !== undefined
    ? latest.mape > 15 ? "critical" : latest.mape > 10 ? "warning" : "normal"
    : "normal";

  const driftBadge = driftStatus === "critical"
    ? "bg-michi-red text-white"
    : driftStatus === "warning"
    ? "bg-michi-amber text-white"
    : "bg-michi-lime text-michi-dark";

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between pb-2">
        <CardTitle className="flex items-center gap-2">
          <Activity size={18} className="text-michi-lime-dark" />
          Simulation Engine
        </CardTitle>
        <Link to="/simulation" className="text-sm text-michi-lime-dark hover:underline font-semibold">
          View details →
        </Link>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-sm text-michi-muted mb-1">Status</p>
            <Badge className={running ? "bg-michi-lime text-michi-dark" : "bg-michi-border text-michi-body"}>
              {running ? "Running" : "Stopped"}
            </Badge>
          </div>
          <div>
            <p className="text-sm text-michi-muted mb-1">Tick</p>
            <p className="text-2xl font-extrabold text-michi-dark">{tick}</p>
          </div>
          <div>
            <p className="text-sm text-michi-muted mb-1">MAPE</p>
            <p className="text-2xl font-extrabold text-michi-dark">
              {latest?.mape !== undefined ? latest.mape.toFixed(1) : "—"}
            </p>
          </div>
          <div>
            <p className="text-sm text-michi-muted mb-1">Drift</p>
            <Badge className={driftBadge}>{driftStatus.toUpperCase()}</Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function CommandCenter() {
  const qc = useQueryClient();
  const { data: suggestionsData } = useQuery({
    queryKey: ["suggestions"],
    queryFn: fetchSuggestions,
    refetchInterval: 30000,
  });
  const { data: interventionsData } = useQuery({
    queryKey: ["interventions"],
    queryFn: () => fetchInterventions(),
    refetchInterval: 15000,
  });
  const updateStatus = useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => updateInterventionStatus(id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["interventions"] }),
  });

  const suggestions = suggestionsData?.suggestions ?? [];
  const interventions = interventionsData?.interventions ?? [];

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Command Center</h1>
        <p className="text-base text-michi-muted mt-1">Real-time overview of Astana bus network operations</p>
      </div>

      <KPIGrid />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CongestionHeatmap />
        <AlertTicker />
      </div>

      <MiniSimulationCard />

      {suggestions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb size={18} className="text-michi-amber" /> Optimization Suggestions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-72 overflow-y-auto">
              {suggestions.slice(0, 8).map((s: Suggestion, i: number) => (
                <div key={i} className={`p-4 rounded-xl border-l-4 ${priorityColor(s.priority)} bg-michi-warm`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant={s.priority === "high" ? "danger" : s.priority === "medium" ? "warning" : "default"}>
                        {s.priority}
                      </Badge>
                      <span className="text-sm font-semibold text-michi-dark">{s.title}</span>
                    </div>
                    <span className="text-xs text-michi-muted font-medium">{s.type}</span>
                  </div>
                  <p className="text-sm text-michi-body mt-1.5">{s.description}</p>
                  {s.action && <p className="text-sm text-michi-lime-dark font-semibold mt-1">{s.action}</p>}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {interventions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap size={18} className="text-michi-lime-dark" /> Active Interventions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2.5 max-h-72 overflow-y-auto">
              {interventions.slice(0, 10).map((intv: Intervention) => (
                <div key={intv.id} className="flex items-center justify-between p-3.5 rounded-xl bg-michi-warm">
                  <div className="flex items-center gap-3">
                    {statusIcon(intv.status)}
                    <div>
                      <div className="text-sm font-semibold text-michi-dark">{intv.intervention_type}</div>
                      <div className="text-xs text-michi-muted">
                        {intv.route_id && `Route ${intv.route_id}`}{intv.station_id && ` · Stn ${intv.station_id}`}
                        {" · "}{intv.status}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    {intv.status === "pending" && (
                      <Button size="sm" variant="default"
                        onClick={() => updateStatus.mutate({ id: intv.id, status: "approved" })}>
                        Approve
                      </Button>
                    )}
                    {intv.status === "approved" && (
                      <Button size="sm" variant="lime"
                        onClick={() => updateStatus.mutate({ id: intv.id, status: "executing" })}>
                        Execute
                      </Button>
                    )}
                    {intv.status === "executing" && (
                      <Button size="sm" variant="outline"
                        onClick={() => updateStatus.mutate({ id: intv.id, status: "completed" })}>
                        Complete
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}