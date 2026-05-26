import { useQuery } from "@tanstack/react-query";
import { fetchPassengerCrowding, fetchServiceChanges, fetchMessagingTemplates } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Users, AlertTriangle, Megaphone } from "lucide-react";

function levelBadge(level: string) {
  const colors: Record<string, string> = {
    low: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300",
    medium: "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300",
    high: "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300",
    very_high: "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300",
    unknown: "bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400",
  };
  return colors[level] || colors.unknown;
}

export default function PassengerPage() {
  const { data: crowding, isLoading: loadingCrowding } = useQuery({
    queryKey: ["passenger-crowding"],
    queryFn: fetchPassengerCrowding,
    refetchInterval: 60000,
  });
  const { data: serviceData } = useQuery({
    queryKey: ["service-changes"],
    queryFn: fetchServiceChanges,
    refetchInterval: 60000,
  });
  const { data: templateData } = useQuery({
    queryKey: ["messaging-templates"],
    queryFn: fetchMessagingTemplates,
  });

  const stations = crowding?.stations ?? [];
  const serviceChanges = (serviceData?.service_changes ?? []) as { severity: string; title: string; message?: string }[];
  const templates = (templateData?.templates ?? []) as { title: string; body: string }[];

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold dark:text-white">Passenger Information</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">Real-time crowding levels and service updates</p>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Users className="h-4 w-4" /> Station Crowding
          </CardTitle>
          <span className="text-xs text-gray-500 dark:text-gray-400">{stations.length} stations</span>
        </CardHeader>
        <CardContent>
          {loadingCrowding ? (
            <p className="text-sm text-gray-400">Loading...</p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 max-h-96 overflow-y-auto">
              {stations.map((s) => (
                <div key={s.station_id} className="p-2 rounded border dark:border-gray-700 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium dark:text-white truncate">{s.name}</span>
                    <span className={`px-1.5 py-0.5 text-[10px] rounded font-medium ${levelBadge(s.current_crowding)}`}>
                      {s.current_crowding}
                    </span>
                  </div>
                  {s.district && <div className="text-[10px] text-gray-400">{s.district}</div>}
                  {s.predictions.length > 0 && (
                    <div className="text-[10px] text-gray-500 dark:text-gray-400">
                      +{s.predictions[0].horizon_minutes}m: {s.predictions[0].level}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {serviceChanges.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-500" /> Service Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {serviceChanges.map((sc, i) => (
                <div key={i} className="p-3 rounded border dark:border-gray-700 bg-amber-50 dark:bg-amber-900/20">
                  <div className="flex items-center gap-2">
                    <span className={`px-1.5 py-0.5 text-[10px] rounded font-medium ${
                      sc.severity === "critical" ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                        : "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                    }`}>{sc.severity}</span>
                    <span className="text-sm font-medium dark:text-white">{sc.title}</span>
                  </div>
                  {sc.message && <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{sc.message}</p>}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {templates.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Megaphone className="h-4 w-4 text-blue-500" /> Messaging Templates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {templates.map((t, i) => (
                <div key={i} className="p-3 rounded border dark:border-gray-700">
                  <div className="text-sm font-medium dark:text-white">{t.title}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{t.body}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}