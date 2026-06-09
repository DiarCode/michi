import { useQuery } from "@tanstack/react-query";
import { fetchDepotStatus, fetchDepotRecommendations } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Truck, Battery, Wrench, AlertTriangle } from "lucide-react";
import type { DepotStatus } from "@/types";
import { CardSkeleton } from "@/components/ui/skeleton";

function DepotCard({ depot }: { depot: DepotStatus["depots"][number] }) {
  const { data: recData } = useQuery({
    queryKey: ["depot-rec", depot.depot_id],
    queryFn: () => fetchDepotRecommendations(depot.depot_id),
    staleTime: 60000,
  });
  const pct = depot.total_buses > 0 ? Math.round((depot.available / depot.total_buses) * 100) : 0;
  const barColor = pct >= 70 ? "bg-michi-lime" : pct >= 40 ? "bg-michi-amber" : "bg-michi-red";
  const barTextColor = pct >= 70 ? "text-michi-lime-dark" : pct >= 40 ? "text-michi-amber" : "text-michi-red";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle>{depot.name}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <Truck size={18} className="mx-auto text-michi-lime-dark mb-1.5" />
            <div className="text-2xl font-extrabold text-michi-dark">{depot.available}</div>
            <div className="text-sm text-michi-muted font-medium">Available</div>
          </div>
          <div>
            <Wrench size={18} className="mx-auto text-michi-amber mb-1.5" />
            <div className="text-2xl font-extrabold text-michi-dark">{depot.maintenance}</div>
            <div className="text-sm text-michi-muted font-medium">Maintenance</div>
          </div>
          <div>
            <Battery size={18} className="mx-auto text-michi-teal mb-1.5" />
            <div className="text-2xl font-extrabold text-michi-dark">{depot.charging}</div>
            <div className="text-sm text-michi-muted font-medium">Charging</div>
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-sm font-medium">
            <span className="text-michi-body">Fleet Utilization</span>
            <span className={barTextColor}>{pct}%</span>
          </div>
          <div className="h-3 bg-michi-warm rounded-full overflow-hidden">
            <div className={`h-3 rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
          </div>
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {depot.routes_served.map((r) => (
            <span key={r} className="px-3 py-1 text-xs bg-michi-lime/15 text-michi-lime-dark rounded-full font-semibold">
              {r}
            </span>
          ))}
        </div>
        {recData?.recommendations && recData.recommendations.length > 0 && (
          <div className="border-t border-michi-border pt-3 space-y-2">
            <div className="text-sm font-semibold text-michi-dark">Recommendations</div>
            {(recData.recommendations as Record<string, string>[]).map((rec, i) => (
              <div key={i} className="text-sm p-2.5 bg-michi-amber/8 border-l-4 border-l-michi-amber rounded-xl flex items-start gap-2">
                <AlertTriangle size={14} className="text-michi-amber flex-shrink-0 mt-0.5" />
                <span className="text-michi-body">{rec.message}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function DepotPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["depot-status"],
    queryFn: fetchDepotStatus,
    refetchInterval: 30000,
  });

  const depots = data?.depots ?? [];

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Depot Operations</h1>
        <p className="text-base text-michi-muted mt-1">Fleet availability and dispatch recommendations</p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <CardSkeleton /><CardSkeleton /><CardSkeleton />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {depots.map((d) => (
            <DepotCard key={d.depot_id} depot={d} />
          ))}
        </div>
      )}

      <Card>
        <CardHeader><CardTitle>Fleet Summary</CardTitle></CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-5 text-center">
            <div>
              <div className="text-3xl font-extrabold text-michi-dark">{depots.reduce((s, d) => s + d.total_buses, 0)}</div>
              <div className="text-sm text-michi-muted font-medium mt-1">Total Fleet</div>
            </div>
            <div>
              <div className="text-3xl font-extrabold text-michi-lime-dark">{depots.reduce((s, d) => s + d.available, 0)}</div>
              <div className="text-sm text-michi-muted font-medium mt-1">Available</div>
            </div>
            <div>
              <div className="text-3xl font-extrabold text-michi-amber">{depots.reduce((s, d) => s + d.maintenance, 0)}</div>
              <div className="text-sm text-michi-muted font-medium mt-1">In Maintenance</div>
            </div>
            <div>
              <div className="text-3xl font-extrabold text-michi-teal">{depots.reduce((s, d) => s + d.charging, 0)}</div>
              <div className="text-sm text-michi-muted font-medium mt-1">Charging</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}