import { useAlerts, useAckAlert } from "@/hooks/useAlerts";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";

export default function AlertsPage() {
  const { data } = useAlerts();
  const ack = useAckAlert();
  const alerts = data?.alerts ?? [];

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Alerts</h2>
      <div className="space-y-3">
        {alerts.map((a: { id: number; severity: string; title: string; message: string }) => (
          <Card key={a.id}>
            <CardContent className="flex items-center gap-4 p-4">
              <AlertTriangle className={a.severity === "high" ? "text-red-500" : "text-amber-500"} />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-semibold">{a.title}</span>
                  <Badge variant={a.severity === "high" ? "danger" : a.severity === "medium" ? "warning" : "default"}>{a.severity}</Badge>
                </div>
                <p className="text-sm text-gray-600">{a.message}</p>
              </div>
              <Button variant="outline" size="sm" onClick={() => ack.mutate(a.id)}>Acknowledge</Button>
            </CardContent>
          </Card>
        ))}
        {alerts.length === 0 && <p className="text-gray-500">No active alerts.</p>}
      </div>
    </div>
  );
}
