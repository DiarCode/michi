import { useAlerts } from "@/hooks/useAlerts";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function AlertTicker() {
  const { data } = useAlerts();
  const alerts = data?.alerts ?? [];

  return (
    <Card>
      <CardHeader><CardTitle className="text-sm">Active Alerts</CardTitle></CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {alerts.map((a) => (
            <div
              key={a.id}
              className={`flex items-start gap-2 p-2 rounded text-xs ${
                a.severity === "high"
                  ? "bg-destructive/10 text-destructive"
                  : a.severity === "medium"
                  ? "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              <Badge variant={a.severity === "high" ? "destructive" : a.severity === "medium" ? "secondary" : "outline"}>
                {a.severity}
              </Badge>
              <div>
                <p className="font-medium">{a.title}</p>
                <p className="opacity-80">{a.message}</p>
              </div>
            </div>
          ))}
          {alerts.length === 0 && <p className="text-muted-foreground text-sm">No active alerts.</p>}
        </div>
      </CardContent>
    </Card>
  );
}
