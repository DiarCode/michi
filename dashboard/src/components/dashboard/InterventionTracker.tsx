import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchInterventions, updateInterventionStatus } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { ZapIcon, CheckmarkCircle01Icon, Clock01Icon, CancelCircleIcon, ArrowRight01Icon } from "@/lib/icons";
import type { Intervention } from "@/types";

const statusSteps = ["pending", "approved", "executing", "completed"] as const;

function statusIcon(status: string) {
  switch (status) {
    case "completed": return <HugeiconsIcon icon={CheckmarkCircle01Icon} className="h-4 w-4 text-chart-2" />;
    case "approved": return <HugeiconsIcon icon={ArrowRight01Icon} className="h-4 w-4 text-chart-1" />;
    case "executing": return <HugeiconsIcon icon={ZapIcon} className="h-4 w-4 text-chart-4" />;
    case "cancelled": return <HugeiconsIcon icon={CancelCircleIcon} className="h-4 w-4 text-muted-foreground" />;
    default: return <HugeiconsIcon icon={Clock01Icon} className="h-4 w-4 text-muted-foreground" />;
  }
}

function statusColor(status: string) {
  switch (status) {
    case "completed": return "bg-chart-2 text-foreground";
    case "approved": return "bg-chart-1 text-white";
    case "executing": return "bg-chart-4 text-white";
    case "cancelled": return "bg-muted text-muted-foreground";
    default: return "bg-muted text-muted-foreground";
  }
}

function StepIndicator({ current, step }: { current: string; step: string }) {
  const idx = statusSteps.indexOf(step as typeof statusSteps[number]);
  const currentIdx = statusSteps.indexOf(current as typeof statusSteps[number]);
  const isComplete = idx < currentIdx;
  const isCurrent = step === current;

  return (
    <div className="flex flex-col items-center gap-0.5">
      <div className={`w-3 h-3 rounded-full border-2 ${
        isComplete ? "bg-chart-2 border-chart-2" :
        isCurrent ? "bg-primary border-primary animate-pulse" :
        "bg-muted border-border"
      }`} />
      <span className={`text-[9px] ${isCurrent ? "font-bold text-foreground" : "text-muted-foreground"}`}>
        {step.charAt(0).toUpperCase() + step.slice(1)}
      </span>
    </div>
  );
}

export default function InterventionTracker() {
  const qc = useQueryClient();
  const { data: interventionsData } = useQuery({
    queryKey: ["interventions"],
    queryFn: () => fetchInterventions(),
    refetchInterval: 15_000,
  });

  const updateStatus = useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => updateInterventionStatus(id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["interventions"] }),
  });

  const interventions = interventionsData?.interventions ?? [];
  if (interventions.length === 0) return null;

  const active = interventions.filter((i: Intervention) => i.status !== "completed" && i.status !== "cancelled");
  const completed = interventions.filter((i: Intervention) => i.status === "completed").slice(0, 3);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <HugeiconsIcon icon={ZapIcon} size={18} className="text-chart-2" />
          Intervention Tracker
          <span className="ml-auto text-xs text-muted-foreground font-normal">
            {active.length} active · {completed.length} completed
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {interventions.slice(0, 8).map((intv: Intervention) => (
            <div key={intv.id} className="p-2.5 rounded-xl bg-muted">
              <div className="flex items-center gap-2">
                {statusIcon(intv.status)}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-semibold text-foreground truncate">
                    {intv.intervention_type.replace(/_/g, " ")}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {intv.route_id && `Route ${intv.route_id}`}{intv.station_id && ` · Stn ${intv.station_id}`}
                  </div>
                </div>
                <Badge className={statusColor(intv.status)}>{intv.status}</Badge>
              </div>

              {/* Step indicator */}
              <div className="flex items-center justify-between mt-2 px-2">
                {statusSteps.map((step) => (
                  <StepIndicator key={step} current={intv.status} step={step} />
                ))}
              </div>

              {/* Action buttons */}
              <div className="flex gap-2 mt-2">
                {intv.status === "pending" && (
                  <Button size="sm" variant="default"
                    onClick={() => updateStatus.mutate({ id: intv.id, status: "approved" })}>
                    Approve
                  </Button>
                )}
                {intv.status === "approved" && (
                  <Button size="sm" variant="default"
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
                {(intv.status === "pending" || intv.status === "approved") && (
                  <Button size="sm" variant="outline"
                    onClick={() => updateStatus.mutate({ id: intv.id, status: "cancelled" })}>
                    Cancel
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function Badge({ className, children }: { className: string; children: React.ReactNode }) {
  return (
    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${className}`}>
      {children}
    </span>
  );
}