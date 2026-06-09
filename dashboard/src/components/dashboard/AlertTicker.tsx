import { useAlerts } from "@/hooks/useAlerts";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

function severityColor(severity: string) {
  if (severity === "high" || severity === "critical") return "bg-michi-red/8 border-l-4 border-l-michi-red";
  if (severity === "medium" || severity === "warning") return "bg-michi-amber/8 border-l-4 border-l-michi-amber";
  return "bg-michi-warm border-l-4 border-l-michi-muted";
}

export default function AlertTicker() {
  const { data } = useAlerts();
  const alerts = data?.alerts ?? [];

  return (
    <Card className="h-full">
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Active Alerts</CardTitle>
        <span className="text-sm text-michi-muted font-medium">{alerts.length} active</span>
      </CardHeader>
      <CardContent>
        <div className="space-y-2.5 max-h-72 overflow-y-auto">
          {alerts.map((a: { id: number; severity: string; title: string; message: string }) => (
            <div key={a.id} className={`flex items-start gap-3 p-3 rounded-xl ${severityColor(a.severity)}`}>
              <Badge variant={a.severity === "high" || a.severity === "critical" ? "danger" : a.severity === "medium" || a.severity === "warning" ? "warning" : "default"}>
                {a.severity}
              </Badge>
              <div className="min-w-0">
                <p className="font-semibold text-sm text-michi-dark">{a.title}</p>
                <p className="text-xs text-michi-body mt-0.5">{a.message}</p>
              </div>
            </div>
          ))}
          {alerts.length === 0 && (
            <div className="text-center py-8">
              <p className="text-base text-michi-muted">No active alerts</p>
              <p className="text-sm text-michi-muted mt-1">All systems operating normally</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}