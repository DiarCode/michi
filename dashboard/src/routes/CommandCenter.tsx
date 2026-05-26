import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import KPIGrid from "@/components/dashboard/KPIGrid";
import CongestionHeatmap from "@/components/dashboard/CongestionHeatmap";
import AlertTicker from "@/components/dashboard/AlertTicker";
import { fetchSuggestions, fetchInterventions, updateInterventionStatus } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Lightbulb, Zap, CheckCircle, Clock, XCircle } from "lucide-react";
import type { Suggestion, Intervention } from "@/types";

function statusIcon(status: string) {
  switch (status) {
    case "completed": return <CheckCircle className="h-4 w-4 text-green-500" />;
    case "approved": return <CheckCircle className="h-4 w-4 text-blue-500" />;
    case "executing": return <Zap className="h-4 w-4 text-amber-500" />;
    case "cancelled": return <XCircle className="h-4 w-4 text-gray-400" />;
    default: return <Clock className="h-4 w-4 text-gray-400" />;
  }
}

function priorityColor(priority: string) {
  if (priority === "critical" || priority === "high") return "border-l-red-500";
  if (priority === "medium") return "border-l-amber-500";
  return "border-l-blue-500";
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
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white">Command Center</h2>
      <KPIGrid />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CongestionHeatmap />
        <AlertTicker />
      </div>

      {suggestions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Lightbulb className="h-4 w-4 text-amber-500" /> Optimization Suggestions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {suggestions.slice(0, 8).map((s: Suggestion, i: number) => (
                <div key={i} className={`p-3 rounded border-l-4 ${priorityColor(s.priority)} bg-gray-50 dark:bg-gray-800`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 text-[10px] rounded font-medium ${
                        s.priority === "high" ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                          : s.priority === "medium" ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                          : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                      }`}>{s.priority}</span>
                      <span className="text-sm font-medium dark:text-white">{s.title}</span>
                    </div>
                    <span className="text-[10px] text-gray-400">{s.type}</span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{s.description}</p>
                  {s.action && <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">{s.action}</p>}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {interventions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Zap className="h-4 w-4 text-blue-500" /> Active Interventions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {interventions.slice(0, 10).map((intv: Intervention) => (
                <div key={intv.id} className="flex items-center justify-between p-2 rounded bg-gray-50 dark:bg-gray-800">
                  <div className="flex items-center gap-2">
                    {statusIcon(intv.status)}
                    <div>
                      <div className="text-sm font-medium dark:text-white">{intv.intervention_type}</div>
                      <div className="text-[10px] text-gray-400">
                        {intv.route_id && `Route ${intv.route_id}`}{intv.station_id && ` · Stn ${intv.station_id}`}
                        {" · "}{intv.status}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-1">
                    {intv.status === "pending" && (
                      <Button size="sm" variant="outline" className="text-xs h-7"
                        onClick={() => updateStatus.mutate({ id: intv.id, status: "approved" })}>
                        Approve
                      </Button>
                    )}
                    {intv.status === "approved" && (
                      <Button size="sm" variant="outline" className="text-xs h-7"
                        onClick={() => updateStatus.mutate({ id: intv.id, status: "executing" })}>
                        Execute
                      </Button>
                    )}
                    {intv.status === "executing" && (
                      <Button size="sm" variant="outline" className="text-xs h-7"
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