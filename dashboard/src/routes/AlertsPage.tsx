import { useState } from "react";
import { useAlerts, useAckAlert } from "@/hooks/useAlerts";
import { api } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";

export default function AlertsPage() {
  const { data } = useAlerts();
  const ack = useAckAlert();
  const [severity, setSeverity] = useState<string>("all");
  const alerts = (data?.alerts ?? []) as Array<{ id: number; severity: string; title: string; message: string; auto?: boolean }>;
  const filtered = severity === "all" ? alerts : alerts.filter((a) => a.severity === severity);

  const generateAlerts = async () => {
    try { await api.post("/alerts/generate"); } catch { /* alert generation is optional */ }
  };

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold">Alerts</h2>
        <div className="flex gap-2">
          <select className="border rounded px-3 py-1.5 text-sm" value={severity} onChange={(e) => setSeverity(e.target.value)}>
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="warning">Warning</option>
            <option value="low">Low</option>
            <option value="info">Info</option>
          </select>
          <Button variant="outline" size="sm" onClick={generateAlerts}>Auto-Generate</Button>
        </div>
      </div>
      <div className="space-y-3">
        {filtered.map((a) => (
          <Card key={a.id}>
            <CardContent className="flex items-center gap-4 p-4">
              <AlertTriangle className={a.severity === "high" || a.severity === "critical" ? "text-red-500" : "text-amber-500"} />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-semibold">{a.title}</span>
                  <Badge variant={a.severity === "high" || a.severity === "critical" ? "danger" : a.severity === "medium" || a.severity === "warning" ? "warning" : "default"}>{a.severity}</Badge>
                  {a.auto && <span className="text-[10px] bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">AUTO</span>}
                </div>
                <p className="text-sm text-gray-600">{a.message}</p>
              </div>
              <Button variant="outline" size="sm" onClick={() => ack.mutate(a.id)}>Acknowledge</Button>
            </CardContent>
          </Card>
        ))}
        {filtered.length === 0 && <p className="text-gray-500">No active alerts.</p>}
      </div>
    </div>
  );
}
