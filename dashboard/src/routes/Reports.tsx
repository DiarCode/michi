import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { useStations } from "@/hooks/useStations";

export default function Reports() {
  const { data } = useStations();
  const stations = data?.stations ?? [];

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Reports</h2>
      <Card>
        <CardHeader><CardTitle>Station Ridership Report</CardTitle></CardHeader>
        <CardContent>
          <table className="w-full text-sm">
            <thead><tr className="border-b"><th className="text-left py-2">Station</th><th className="text-right py-2">District</th><th className="text-right py-2">24h Ridership</th></tr></thead>
            <tbody>
              {stations.map((s: { id: string; name: string; district: string; ridership_24h: number }) => (
                <tr key={s.id} className="border-b hover:bg-gray-50"><td className="py-2">{s.name}</td><td className="text-right py-2">{s.district ?? "—"}</td><td className="text-right py-2 font-mono">{s.ridership_24h?.toLocaleString() ?? "—"}</td></tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}
