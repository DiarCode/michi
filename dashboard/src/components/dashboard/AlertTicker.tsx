import { useAlerts } from "@/hooks/useAlerts";
import { Badge } from "@/components/ui/badge";
import { severityColor } from "@/lib/utils";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function AlertTicker() {
  const { data } = useAlerts();
  const alerts = data?.alerts ?? [];

  return (
    <Card>
      <CardHeader><CardTitle className="text-sm">Active Alerts</CardTitle></CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {alerts.map((a: { id: number; severity: string; title: string; message: string }) => (
            <div key={a.id} className={`flex items-start gap-2 p-2 rounded ${severityColor(a.severity)}`}>
              <Badge variant={a.severity === "high" ? "danger" : a.severity === "medium" ? "warning" : "default"}>{a.severity}</Badge>
              <div><p className="font-medium text-sm">{a.title}</p><p className="text-xs opacity-80">{a.message}</p></div>
            </div>
          ))}
          {alerts.length === 0 && <p className="text-gray-400 text-sm">No active alerts.</p>}
        </div>
      </CardContent>
    </Card>
  );
}
