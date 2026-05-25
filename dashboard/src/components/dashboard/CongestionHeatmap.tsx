import { useStations } from "@/hooks/useStations";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function CongestionHeatmap() {
  const { data } = useStations();
  const stations = data?.stations ?? [];
  const maxRidership = Math.max(...stations.map((s) => s.ridership_24h ?? 0), 1);

  return (
    <Card>
      <CardHeader><CardTitle className="text-sm">Congestion Heatmap</CardTitle></CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-2">
          {stations.map((s) => {
            const intensity = (s.ridership_24h ?? 0) / maxRidership;
            const bg = intensity > 0.7 ? "bg-red-400" : intensity > 0.4 ? "bg-amber-400" : "bg-green-400";
            return <div key={s.id} className={`p-2 rounded text-xs text-center text-white ${bg}`} title={`${s.name}: ${s.ridership_24h ?? 0}`}>{s.name.split(" ").pop()}</div>;
          })}
        </div>
      </CardContent>
    </Card>
  );
}
