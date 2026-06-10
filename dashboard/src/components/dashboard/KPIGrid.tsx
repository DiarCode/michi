import { useQuery } from "@tanstack/react-query";
import { fetchKPIs } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function KPIGrid() {
  const { data: kpis } = useQuery({ queryKey: ["kpis"], queryFn: fetchKPIs, refetchInterval: 30000 });
  const items = [
    { label: "Stations", value: kpis?.total_stations ?? "—" },
    { label: "Routes", value: kpis?.active_routes ?? "—" },
    { label: "Avg Ridership", value: kpis?.avg_ridership ?? "—" },
    { label: "On-Time", value: kpis?.on_time_performance ? `${kpis.on_time_performance}%` : "—" },
    { label: "Alerts", value: kpis?.alerts_today ?? "—" },
    { label: "Peak Hour", value: kpis?.peak_hour ?? "—" },
  ];

  return (
    <div className="grid grid-cols-3 lg:grid-cols-6 gap-4">
      {items.map((item) => (
        <Card key={item.label}>
          <CardHeader><CardTitle className="text-xs text-muted-foreground">{item.label}</CardTitle></CardHeader>
          <CardContent><div className="text-2xl font-bold">{item.value}</div></CardContent>
        </Card>
      ))}
    </div>
  );
}
