import { useState } from "react";
import { useAlerts, useAckAlert } from "@/hooks/useAlerts";
import { api } from "@/lib/api";
import { showToast } from "@/lib/toast";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { Alert01Icon, CheckmarkCircle02Icon } from "@/lib/icons";
import { ListSkeleton } from "@/components/ui/skeleton";

const SEVERITY_OPTIONS = [
  { value: "all", label: "All" },
  { value: "critical", label: "Critical" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "warning", label: "Warning" },
  { value: "low", label: "Low" },
  { value: "info", label: "Info" },
];

export default function AlertsPage() {
  const { data, isLoading } = useAlerts();
  const ack = useAckAlert();
  const [severity, setSeverity] = useState<string>("all");
  const alerts = (data?.alerts ?? []) as Array<{ id: number; severity: string; title: string; message: string; auto?: boolean }>;
  const filtered = severity === "all" ? alerts : alerts.filter((a) => a.severity === severity);

  const generateAlerts = async () => {
    try { await api.post("/alerts/generate"); }
    catch (err: any) { showToast.error(`Failed to generate alerts: ${err.message}`); }
  };

  if (isLoading) return <div className="p-8"><ListSkeleton count={3} /></div>;

  const criticalCount = alerts.filter(a => a.severity === "critical" || a.severity === "high").length;

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-foreground">Alerts</h1>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-muted-foreground font-medium">Total Alerts</span>
            <p className="text-3xl font-extrabold text-foreground mt-2">{alerts.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-muted-foreground font-medium">Critical / High</span>
            <p className="text-3xl font-extrabold text-destructive mt-2">{criticalCount}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-muted-foreground font-medium">Auto-Generated</span>
            <p className="text-3xl font-extrabold text-foreground mt-2">{alerts.filter(a => a.auto).length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-muted-foreground font-medium">Severity Filter</span>
            <p className="text-3xl font-extrabold text-chart-2 mt-2 capitalize">{severity === "all" ? "Showing All" : severity}</p>
          </CardContent>
        </Card>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex gap-2 flex-wrap">
          {SEVERITY_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setSeverity(opt.value)}
              className={`px-4 py-2 text-sm rounded-full font-semibold transition-all ${
                severity === opt.value
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-card border border-border text-muted-foreground hover:bg-muted"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <Button variant="outline" size="sm" onClick={generateAlerts}>Auto-Generate</Button>
      </div>

      <div className="space-y-3">
        {filtered.map((a) => {
          const isHigh = a.severity === "high" || a.severity === "critical";
          const isMedium = a.severity === "medium" || a.severity === "warning";
          const borderColor = isHigh ? "border-l-destructive" : isMedium ? "border-l-chart-4" : "border-l-muted-foreground";
          return (
            <Card key={a.id}>
              <CardContent className={`flex items-center gap-4 p-5 border-l-4 ${borderColor}`}>
                <HugeiconsIcon icon={Alert01Icon} size={20} className={isHigh ? "text-destructive shrink-0" : "text-chart-4 shrink-0"} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-foreground">{a.title}</span>
                    <Badge variant={isHigh ? "destructive" : isMedium ? "secondary" : "default"}>{a.severity}</Badge>
                    {a.auto && (
                      <span className="text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded-full font-semibold border border-border">AUTO</span>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">{a.message}</p>
                </div>
                <Button variant="outline" size="sm" onClick={() => ack.mutate(a.id)}>
                  <HugeiconsIcon icon={CheckmarkCircle02Icon} size={14} className="mr-1" />
                  Acknowledge
                </Button>
              </CardContent>
            </Card>
          );
        })}
        {filtered.length === 0 && (
          <Card>
            <CardContent className="text-center py-12">
              <HugeiconsIcon icon={CheckmarkCircle02Icon} size={32} className="text-chart-2 mx-auto mb-3" />
              <p className="text-lg font-semibold text-foreground">No active alerts</p>
              <p className="text-sm text-muted-foreground mt-1">All systems operating normally</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}