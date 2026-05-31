import { useQuery } from "@tanstack/react-query";
import { fetchNetworkGraph } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { GitGraph, MapPin, Route } from "lucide-react";
import { GridSkeleton, ChartSkeleton } from "@/components/ui/skeleton";

export default function NetworkPage() {
  const { data: graph, isLoading } = useQuery({ queryKey: ["network-graph"], queryFn: fetchNetworkGraph });

  if (isLoading) return <div className="p-6 space-y-6"><GridSkeleton /><ChartSkeleton /></div>;

  const nodes = graph?.nodes ?? [];
  const edges = graph?.edges ?? [];
  const districts = graph?.districts ?? {};
  const stats = graph?.stats ?? { total_stations: 0, total_routes: 0, total_edges: 0 };

  const districtColors: Record<string, string> = {
    Esil: "bg-blue-500",
    Almaty: "bg-emerald-500",
    Saryarka: "bg-amber-500",
    Baikonur: "bg-purple-500",
    Unknown: "bg-gray-400",
  };

  const latRange = nodes.length ? { min: Math.min(...nodes.map((n) => n.lat)), max: Math.max(...nodes.map((n) => n.lat)) } : { min: 0, max: 1 };
  const lonRange = nodes.length ? { min: Math.min(...nodes.map((n) => n.lon)), max: Math.max(...nodes.map((n) => n.lon)) } : { min: 0, max: 1 };

  const toX = (lon: number) => ((lon - lonRange.min) / (lonRange.max - lonRange.min || 1)) * 100;
  const toY = (lat: number) => ((latRange.max - lat) / (latRange.max - latRange.min || 1)) * 100;

  const svgColor: Record<string, string> = {
    Esil: "#3b82f6", Almaty: "#10b981", Saryarka: "#f59e0b", Baikonur: "#8b5cf6", Unknown: "#9ca3af",
  };

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Network Topology</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400">Visualize the transit network graph, adjacency, and district coverage.</p>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Stations</CardTitle>
            <MapPin className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{stats.total_stations}</p></CardContent>
        </Card>
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Routes</CardTitle>
            <Route className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{stats.total_routes}</p></CardContent>
        </Card>
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Connections</CardTitle>
            <GitGraph className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{stats.total_edges}</p></CardContent>
        </Card>
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Districts</CardTitle>
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{Object.keys(districts).length}</p></CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Network Graph</CardTitle></CardHeader>
          <CardContent>
            <div className="relative w-full h-96 bg-gray-100 dark:bg-gray-800 rounded-lg overflow-hidden">
              <svg viewBox="0 0 100 100" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
                {edges.map((e, i) => {
                  const fromNode = nodes.find((n) => n.id === e.from);
                  const toNode = nodes.find((n) => n.id === e.to);
                  if (!fromNode || !toNode) return null;
                  return (
                    <line key={i} x1={toX(fromNode.lon)} y1={toY(fromNode.lat)} x2={toX(toNode.lon)} y2={toY(toNode.lat)} stroke="#94a3b8" strokeWidth="0.3" />
                  );
                })}
                {nodes.map((n) => (
                  <circle key={n.id} cx={toX(n.lon)} cy={toY(n.lat)} r="1.2" fill={svgColor[n.district] ?? "#9ca3af"} />
                ))}
              </svg>
            </div>
            <div className="flex gap-4 mt-3 text-xs">
              {Object.entries(districtColors).map(([name, color]) => (
                <span key={name} className="flex items-center gap-1"><span className={`w-3 h-3 rounded-full ${color}`} /> {name} ({districts[name] ?? 0})</span>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>District Breakdown</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(districts).sort((a, b) => b[1] - a[1]).map(([name, count]) => {
                const pct = Math.round((count / stats.total_stations) * 100);
                return (
                  <div key={name}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium">{name}</span>
                      <span className="text-gray-500 dark:text-gray-400">{count} stations ({pct}%)</span>
                    </div>
                    <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${districtColors[name] ?? "bg-gray-400"}`} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Station List</CardTitle></CardHeader>
        <CardContent>
          <div className="max-h-64 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white dark:bg-gray-900">
                <tr className="border-b">
                  <th className="text-left py-2">ID</th>
                  <th className="text-left py-2">Name</th>
                  <th className="text-left py-2">District</th>
                  <th className="text-right py-2">Lat</th>
                  <th className="text-right py-2">Lon</th>
                </tr>
              </thead>
              <tbody>
                {nodes.slice(0, 50).map((n) => (
                  <tr key={n.id} className="border-b hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="py-1.5 font-mono text-xs">{n.id}</td>
                    <td className="py-1.5">{n.name}</td>
                    <td className="py-1.5"><span className={`px-1.5 py-0.5 rounded text-xs text-white ${districtColors[n.district] ?? "bg-gray-400"}`}>{n.district}</span></td>
                    <td className="py-1.5 text-right font-mono text-xs">{n.lat.toFixed(4)}</td>
                    <td className="py-1.5 text-right font-mono text-xs">{n.lon.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {nodes.length > 50 && <p className="text-xs text-gray-400 mt-2">Showing first 50 of {nodes.length} stations</p>}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}