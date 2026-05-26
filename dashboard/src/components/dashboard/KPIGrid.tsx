import { useQuery } from "@tanstack/react-query";
import { fetchKPIs } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { MapPin, Route, Users, Clock, AlertTriangle, TrendingUp } from "lucide-react";

const ICONS: Record<string, React.ElementType> = {
  Stations: MapPin, Routes: Route, Ridership: TrendingUp,
  "On-Time": Clock, Alerts: AlertTriangle, "Peak Hour": Users,
};

export default function KPIGrid() {
  const { data: kpis } = useQuery({ queryKey: ["kpis"], queryFn: fetchKPIs, refetchInterval: 30000 });
  const items = [
    { label: "Stations", value: kpis?.total_stations ?? "—", icon: "Stations" },
    { label: "Routes", value: kpis?.active_routes ?? "—", icon: "Routes" },
    { label: "Ridership", value: kpis?.avg_ridership != null ? Math.round(kpis.avg_ridership).toLocaleString() : "—", icon: "Ridership" },
    { label: "On-Time", value: kpis?.on_time_performance ? `${kpis.on_time_performance}%` : "—", icon: "On-Time" },
    { label: "Alerts", value: kpis?.alerts_today ?? "—", icon: "Alerts" },
    { label: "Peak Hour", value: kpis?.peak_hour ?? "—", icon: "Peak Hour" },
  ];

  return (
    <div className="grid grid-cols-3 lg:grid-cols-6 gap-4">
      {items.map((item) => {
        const Icon = ICONS[item.icon] ?? MapPin;
        return (
          <Card key={item.label} className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">{item.label}</CardTitle>
              <Icon className="h-4 w-4 text-gray-400 dark:text-gray-500" />
            </CardHeader>
            <CardContent><div className="text-2xl font-bold tracking-tight">{item.value}</div></CardContent>
          </Card>
        );
      })}
    </div>
  );
}
