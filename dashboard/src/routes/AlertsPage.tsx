import { useQuery } from "@tanstack/react-query";
import { fetchAlerts } from "../lib/api";

export default function AlertsPage() {
  const { data } = useQuery({ queryKey: ["alerts"], queryFn: fetchAlerts });
  const alerts = data?.alerts ?? [];
  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Alerts</h2>
      <div className="space-y-3">
        {alerts.map((alert: any) => (
          <div key={alert.id} className="bg-white p-4 rounded shadow flex items-center gap-4">
            <div className="text-red-500 font-bold">!</div>
            <div>
              <div className="font-semibold">{alert.alert_type}</div>
              <div className="text-sm text-gray-600">{alert.message}</div>
            </div>
          </div>
        ))}
        {alerts.length === 0 && <p className="text-gray-500">No alerts.</p>}
      </div>
    </div>
  );
}
