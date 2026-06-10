import { useConnectionProtection } from "@/hooks/useConnectionProtection";
import { formatEta } from "@/lib/connectionProtection";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { HugeiconsIcon } from "@hugeicons/react";
import { Shield01Icon, ShieldQuestionMarkIcon, Clock01Icon } from "@/lib/icons";

const riskIcons = {
  at_risk: <HugeiconsIcon icon={Shield01Icon} className="h-4 w-4 text-destructive" />,
  tight: <HugeiconsIcon icon={ShieldQuestionMarkIcon} className="h-4 w-4 text-chart-4" />,
  safe: <HugeiconsIcon icon={Shield01Icon} className="h-4 w-4 text-chart-2" />,
} as const;

const riskBg = {
  at_risk: "bg-destructive/10 border-l-destructive",
  tight: "bg-chart-4/10 border-l-chart-4",
  safe: "bg-chart-2/10 border-l-chart-2",
} as const;

export default function ConnectionProtectionPanel() {
  const { risks, summary } = useConnectionProtection();

  if (risks.length === 0) return null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <HugeiconsIcon icon={Shield01Icon} size={18} className="text-chart-2" />
          Connection Protection
          <span className="ml-auto flex items-center gap-2 text-xs font-normal text-muted-foreground">
            <span className="text-destructive font-semibold">{summary.atRisk}</span> at risk
            <span className="text-chart-4 font-semibold">{summary.tight}</span> tight
            <span className="text-chart-2 font-semibold">{summary.safe}</span> safe
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {risks.slice(0, 10).map((r) => (
            <div
              key={`${r.arriving.busId}-${r.departing.busId}-${r.stationId}`}
              className={`p-2.5 rounded-lg border-l-4 ${riskBg[r.riskLevel]} text-xs`}
            >
              <div className="flex items-center gap-1.5 font-semibold">
                {riskIcons[r.riskLevel]}
                <span className="flex-1 truncate">{r.stationName}</span>
                <span className={`font-mono font-bold ${
                  r.riskLevel === "at_risk" ? "text-destructive" :
                  r.riskLevel === "tight" ? "text-chart-4" :
                  "text-chart-2"
                }`}>
                  {r.transferWindowSec < 0
                    ? `Missed by ${formatEta(Math.abs(r.transferWindowSec))}`
                    : `${formatEta(r.transferWindowSec)} window`}
                </span>
              </div>
              <div className="flex items-center gap-3 mt-1 text-muted-foreground">
                <span className="flex items-center gap-1">
                  <HugeiconsIcon icon={Clock01Icon} size={10} />
                  Arriving: <strong className="text-muted-foreground">Route {r.arriving.routeId}</strong>
                  <span className="font-mono">{formatEta(r.arriving.etaSeconds)}</span>
                </span>
                <span className="flex items-center gap-1">
                  → Departing: <strong className="text-muted-foreground">Route {r.departing.routeId}</strong>
                  <span className="font-mono">{formatEta(r.departing.etaSeconds)}</span>
                </span>
              </div>
              {r.estimatedPassengers > 0 && (
                <div className="mt-1 text-muted-foreground">
                  ~{r.estimatedPassengers} passengers affected
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}