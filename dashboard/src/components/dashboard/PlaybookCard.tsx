import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { createIntervention, simulateIntervention } from "@/lib/api";
import { getPlaybookForFamily, getPlaybookForActions } from "@/lib/playbooks";
import type { RichAlert } from "@/types";
import ConfidenceBadge from "@/components/ui/ConfidenceBadge";

interface PlaybookCardProps {
  alert: RichAlert;
  /** Optional route/station context */
  routeId?: string;
  stationId?: string;
}

export default function PlaybookCard({ alert, routeId, stationId }: PlaybookCardProps) {
  const qc = useQueryClient();
  const [simulating, setSimulating] = useState<string | null>(null);
  const [simulatedImpact, setSimulatedImpact] = useState<Record<string, Record<string, unknown>>>({});
  const [activating, setActivating] = useState<string | null>(null);

  // Find matching playbook
  const playbook = alert.family
    ? getPlaybookForFamily(alert.family)
    : alert.recommended_actions?.length
      ? getPlaybookForActions(alert.recommended_actions.map((a) => a.type))
      : undefined;

  if (!playbook) return null;

  const handleSimulate = async (step: typeof playbook.steps[0]) => {
    setSimulating(step.interventionType);
    try {
      const result = await simulateIntervention(
        step.interventionType,
        routeId ?? alert.route_id,
        stationId ?? alert.station_id,
      );
      setSimulatedImpact((prev) => ({ ...prev, [step.interventionType]: result }));
    } catch {
      // Silently fail — simulation is optional
    } finally {
      setSimulating(null);
    }
  };

  const handleActivate = async (step: typeof playbook.steps[0]) => {
    setActivating(step.interventionType);
    try {
      await createIntervention({
        alert_id: alert.id,
        intervention_type: step.interventionType,
        route_id: routeId ?? alert.route_id,
        station_id: stationId ?? alert.station_id,
      });
      qc.invalidateQueries({ queryKey: ["interventions"] });
    } catch {
      // Error handling via toast could be added
    } finally {
      setActivating(null);
    }
  };

  return (
    <div className="p-3 rounded-xl border border-border bg-card">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">{playbook.icon}</span>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-bold text-foreground truncate">{playbook.title}</h4>
          <p className="text-xs text-muted-foreground">{playbook.description}</p>
        </div>
        {alert.confidence != null && (
          <ConfidenceBadge confidence={alert.confidence} compact />
        )}
      </div>

      <div className="space-y-2">
        {playbook.steps.map((step, i) => {
          const impact = simulatedImpact[step.interventionType];
          const isSimulating = simulating === step.interventionType;
          const isActivating = activating === step.interventionType;

          return (
            <div
              key={step.interventionType}
              className="flex items-start gap-2 p-2 rounded-lg bg-muted"
            >
              <span className="text-sm mt-0.5">{step.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-semibold text-foreground">{step.label}</span>
                  <span className="text-[10px] text-muted-foreground">Step {i + 1}</span>
                </div>
                <p className="text-[11px] text-muted-foreground mt-0.5">{step.expectedImpact}</p>
                {step.slaMinutes && (
                  <span className="text-[10px] text-muted-foreground font-mono">⏱ SLA: {step.slaMinutes}m</span>
                )}
                {impact && (
                  <div className="mt-1 text-[10px] text-chart-2 font-mono">
                    {impact.ridership_change != null && (
                      <span>Ridership: {Number(impact.ridership_change) > 0 ? "+" : ""}{String(impact.ridership_change)}% </span>
                    )}
                    {impact.wait_time_change != null && (
                      <span>Wait: {Number(impact.wait_time_change) > 0 ? "+" : ""}{String(impact.wait_time_change)}min</span>
                    )}
                  </div>
                )}
              </div>
              <div className="flex flex-col gap-1">
                <button
                  onClick={() => handleSimulate(step)}
                  disabled={isSimulating}
                  className="px-2 py-0.5 text-[10px] rounded font-semibold bg-chart-2/20 text-chart-2 hover:bg-chart-2/40 disabled:opacity-50 transition-colors"
                >
                  {isSimulating ? "..." : "Simulate"}
                </button>
                <button
                  onClick={() => handleActivate(step)}
                  disabled={isActivating}
                  className="px-2 py-0.5 text-[10px] rounded font-semibold bg-primary text-primary-foreground hover:bg-primary/80 disabled:opacity-50 transition-colors"
                >
                  {isActivating ? "..." : "Activate"}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {alert.consequence_if_ignored && (
        <div className="mt-2 p-2 rounded-lg bg-destructive/10 text-xs text-destructive">
          ⚠ If ignored: {alert.consequence_if_ignored}
        </div>
      )}
    </div>
  );
}