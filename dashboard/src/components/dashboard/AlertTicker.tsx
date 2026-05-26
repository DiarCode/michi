import { useAlerts } from "@/hooks/useAlerts";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

function severityColor(severity: string) {
  if (severity === "high") return "bg-red-50 dark:bg-red-900/30 text-red-800 dark:text-red-300 border-red-200 dark:border-red-800";
  if (severity === "medium") return "bg-amber-50 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 border-amber-200 dark:border-amber-800";
  return "bg-blue-50 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 border-blue-200 dark:border-blue-800";
}

export default function AlertTicker() {
  const { data } = useAlerts();
  const alerts = data?.alerts ?? [];

  return (
    <Card>
      <CardHeader><CardTitle className="text-sm">Active Alerts</CardTitle></CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {alerts.map((a: { id: number; severity: string; title: string; message: string }) => (
            <div key={a.id} className={`flex items-start gap-2 p-2 rounded border ${severityColor(a.severity)}`}>
              <Badge variant={a.severity === "high" ? "danger" : a.severity === "medium" ? "warning" : "default"}>{a.severity}</Badge>
              <div><p className="font-medium text-sm">{a.title}</p><p className="text-xs opacity-80">{a.message}</p></div>
            </div>
          ))}
          {alerts.length === 0 && <p className="text-gray-400 dark:text-gray-500 text-sm">No active alerts.</p>}
        </div>
      </CardContent>
    </Card>
  );
}
