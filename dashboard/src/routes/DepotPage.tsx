import { useQuery } from "@tanstack/react-query";
import { fetchDepotStatus, fetchDepotRecommendations } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Truck, Battery, Wrench, AlertTriangle } from "lucide-react";
import type { DepotStatus } from "@/types";

function DepotCard({ depot }: { depot: DepotStatus["depots"][number] }) {
  const { data: recData } = useQuery({
    queryKey: ["depot-rec", depot.depot_id],
    queryFn: () => fetchDepotRecommendations(depot.depot_id),
    staleTime: 60000,
  });
  const pct = depot.total_buses > 0 ? Math.round((depot.available / depot.total_buses) * 100) : 0;
  const barColor = pct >= 70 ? "bg-green-500" : pct >= 40 ? "bg-amber-500" : "bg-red-500";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{depot.name}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-3 gap-3 text-center">
          <div>
            <Truck className="h-4 w-4 mx-auto text-blue-500" />
            <div className="text-lg font-bold">{depot.available}</div>
            <div className="text-[10px] text-gray-500 dark:text-gray-400">Available</div>
          </div>
          <div>
            <Wrench className="h-4 w-4 mx-auto text-amber-500" />
            <div className="text-lg font-bold">{depot.maintenance}</div>
            <div className="text-[10px] text-gray-500 dark:text-gray-400">Maintenance</div>
          </div>
          <div>
            <Battery className="h-4 w-4 mx-auto text-green-500" />
            <div className="text-lg font-bold">{depot.charging}</div>
            <div className="text-[10px] text-gray-500 dark:text-gray-400">Charging</div>
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>Fleet Utilization</span>
            <span>{pct}%</span>
          </div>
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
            <div className={`h-2 rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
          </div>
        </div>
        <div className="flex gap-1 flex-wrap">
          {depot.routes_served.map((r) => (
            <span key={r} className="px-1.5 py-0.5 text-[10px] bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
              {r}
            </span>
          ))}
        </div>
        {recData?.recommendations && recData.recommendations.length > 0 && (
          <div className="border-t dark:border-gray-700 pt-2 space-y-1">
            <div className="text-xs font-semibold dark:text-gray-300">Recommendations</div>
            {(recData.recommendations as Record<string, string>[]).map((rec, i) => (
              <div key={i} className="text-xs p-1.5 bg-amber-50 dark:bg-amber-900/20 rounded flex items-start gap-1">
                <AlertTriangle className="h-3 w-3 text-amber-500 flex-shrink-0 mt-0.5" />
                <span className="dark:text-gray-300">{rec.message}</span>
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
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold dark:text-white">Depot Operations</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">Fleet availability and dispatch recommendations</p>
      </div>

      {isLoading ? (
        <p className="text-gray-400 dark:text-gray-500">Loading depot data...</p>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {depots.map((d) => (
            <DepotCard key={d.depot_id} depot={d} />
          ))}
        </div>
      )}

      <Card>
        <CardHeader><CardTitle className="text-sm">Fleet Summary</CardTitle></CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold dark:text-white">{depots.reduce((s, d) => s + d.total_buses, 0)}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Total Fleet</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">{depots.reduce((s, d) => s + d.available, 0)}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Available</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-amber-600">{depots.reduce((s, d) => s + d.maintenance, 0)}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">In Maintenance</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">{depots.reduce((s, d) => s + d.charging, 0)}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Charging</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}