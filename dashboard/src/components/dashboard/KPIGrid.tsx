import { useQuery } from "@tanstack/react-query";
import { fetchKPIs } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
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
    <div className="grid grid-cols-3 lg:grid-cols-6 gap-5">
      {items.map((item) => {
        const Icon = ICONS[item.icon] ?? MapPin;
        return (
          <Card key={item.label} className="hover:shadow-card-hover transition-shadow">
            <CardContent className="p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-michi-muted">{item.label}</span>
                <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                  <Icon size={16} className="text-michi-lime-dark" />
                </div>
              </div>
              <div className="text-3xl font-extrabold text-michi-dark tracking-tight">{item.value}</div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}