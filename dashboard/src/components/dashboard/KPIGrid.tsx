import { useQuery } from "@tanstack/react-query";
import { fetchKPIs } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { HugeiconsIcon } from "@hugeicons/react";
import { MapPinIcon, RouteIcon, UserMultipleIcon, Clock01Icon, Alert01Icon, ArrowUp01Icon } from "@/lib/icons";

const ICONS: Record<string, any> = {
  Stations: MapPinIcon, Routes: RouteIcon, Ridership: ArrowUp01Icon,
  "On-Time": Clock01Icon, Alerts: Alert01Icon, "Peak Hour": UserMultipleIcon,
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
        const icon = ICONS[item.icon] ?? MapPinIcon;
        return (
          <Card key={item.label} className="hover:shadow-md transition-shadow">
            <CardContent className="p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-muted-foreground">{item.label}</span>
                <div className="w-8 h-8 rounded-full bg-chart-2/15 flex items-center justify-center">
                  <HugeiconsIcon icon={icon} size={16} className="text-chart-2" />
                </div>
              </div>
              <div className="text-3xl font-extrabold text-foreground tracking-tight">{item.value}</div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}