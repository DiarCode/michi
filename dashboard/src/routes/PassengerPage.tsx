import { useQuery } from "@tanstack/react-query";
import { fetchPassengerCrowding, fetchServiceChanges, fetchMessagingTemplates } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Users, AlertTriangle, Megaphone } from "lucide-react";
import { GridSkeleton } from "@/components/ui/skeleton";

function levelBadge(level: string) {
  const colors: Record<string, string> = {
    low: "bg-michi-lime/15 text-michi-lime-dark",
    medium: "bg-michi-amber/15 text-michi-amber",
    high: "bg-orange-100 text-orange-700",
    very_high: "bg-michi-red/10 text-michi-red",
    unknown: "bg-michi-warm text-michi-muted",
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
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Passenger Information</h1>
        <p className="text-base text-michi-muted mt-1">Real-time crowding levels and service updates</p>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Users size={18} className="text-michi-lime-dark" />
            Station Crowding
          </CardTitle>
          <span className="text-sm text-michi-muted font-medium">{stations.length} stations</span>
        </CardHeader>
        <CardContent>
          {loadingCrowding ? (
            <GridSkeleton count={3} />
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 max-h-96 overflow-y-auto">
              {stations.map((s) => (
                <div key={s.station_id} className="p-3.5 rounded-xl border border-michi-border space-y-1.5 hover:shadow-card-hover transition-shadow">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-michi-dark truncate">{s.name}</span>
                    <span className={`px-2.5 py-1 text-xs rounded-full font-semibold ${levelBadge(s.current_crowding)}`}>
                      {s.current_crowding}
                    </span>
                  </div>
                  {s.district && <div className="text-xs text-michi-muted font-medium">{s.district}</div>}
                  {s.predictions.length > 0 && (
                    <div className="text-xs text-michi-body font-medium">
                      +{s.predictions[0].horizon_minutes}m: <span className="capitalize">{s.predictions[0].level}</span>
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
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle size={18} className="text-michi-amber" /> Service Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2.5 max-h-72 overflow-y-auto">
              {serviceChanges.map((sc, i) => (
                <div key={i} className="p-4 rounded-xl border border-michi-border bg-michi-amber/5 border-l-4 border-l-michi-amber">
                  <div className="flex items-center gap-2">
                    <span className={`px-2.5 py-1 text-xs rounded-full font-semibold ${
                      sc.severity === "critical" ? "bg-michi-red/10 text-michi-red" : "bg-michi-amber/15 text-michi-amber"
                    }`}>{sc.severity}</span>
                    <span className="text-sm font-semibold text-michi-dark">{sc.title}</span>
                  </div>
                  {sc.message && <p className="text-sm text-michi-body mt-1.5">{sc.message}</p>}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {templates.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Megaphone size={18} className="text-michi-lime-dark" /> Messaging Templates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2.5">
              {templates.map((t, i) => (
                <div key={i} className="p-4 rounded-xl border border-michi-border hover:shadow-card-hover transition-shadow">
                  <div className="text-sm font-semibold text-michi-dark">{t.title}</div>
                  <div className="text-sm text-michi-muted mt-1.5">{t.body}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}