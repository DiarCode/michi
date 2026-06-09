import { useQuery } from "@tanstack/react-query";
import { fetchNetworkGraph } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { GitGraph, MapPin, Route } from "lucide-react";
import { GridSkeleton, ChartSkeleton } from "@/components/ui/skeleton";

export default function NetworkPage() {
  const { data: graph, isLoading } = useQuery({ queryKey: ["network-graph"], queryFn: fetchNetworkGraph });

  if (isLoading) return <div className="p-8 space-y-8"><GridSkeleton /><ChartSkeleton /></div>;

  const nodes = graph?.nodes ?? [];
  const edges = graph?.edges ?? [];
  const districts = graph?.districts ?? {};
  const stats = graph?.stats ?? { total_stations: 0, total_routes: 0, total_edges: 0 };

  const districtColors: Record<string, string> = {
    Esil: "bg-michi-lime",
    Almaty: "bg-michi-teal",
    Saryarka: "bg-michi-amber",
    Baikonur: "bg-michi-purple",
    Unknown: "bg-michi-muted",
  };

  const latRange = nodes.length ? { min: Math.min(...nodes.map((n) => n.lat)), max: Math.max(...nodes.map((n) => n.lat)) } : { min: 0, max: 1 };
  const lonRange = nodes.length ? { min: Math.min(...nodes.map((n) => n.lon)), max: Math.max(...nodes.map((n) => n.lon)) } : { min: 0, max: 1 };

  const toX = (lon: number) => ((lon - lonRange.min) / (lonRange.max - lonRange.min || 1)) * 100;
  const toY = (lat: number) => ((latRange.max - lat) / (latRange.max - latRange.min || 1)) * 100;

  const svgColor: Record<string, string> = {
    Esil: "#B1E743", Almaty: "#2ABFBF", Saryarka: "#F5A623", Baikonur: "#8B5CF6", Unknown: "#9C9C95",
  };

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Network Topology</h1>
        <p className="text-base text-michi-muted mt-1">Transit graph, adjacency, and district coverage</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-michi-muted font-medium">Stations</span>
              <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                <MapPin size={16} className="text-michi-lime-dark" />
              </div>
            </div>
            <p className="text-3xl font-extrabold text-michi-dark">{stats.total_stations}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-michi-muted font-medium">Routes</span>
              <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                <Route size={16} className="text-michi-lime-dark" />
              </div>
            </div>
            <p className="text-3xl font-extrabold text-michi-dark">{stats.total_routes}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-michi-muted font-medium">Connections</span>
              <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                <GitGraph size={16} className="text-michi-lime-dark" />
              </div>
            </div>
            <p className="text-3xl font-extrabold text-michi-dark">{stats.total_edges}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-michi-muted font-medium">Districts</span>
            <p className="text-3xl font-extrabold text-michi-dark mt-2">{Object.keys(districts).length}</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Network Graph</CardTitle></CardHeader>
          <CardContent>
            <div className="relative w-full h-96 bg-michi-warm rounded-xl overflow-hidden">
              <svg viewBox="0 0 100 100" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
                {edges.map((e, i) => {
                  const fromNode = nodes.find((n) => n.id === e.from);
                  const toNode = nodes.find((n) => n.id === e.to);
                  if (!fromNode || !toNode) return null;
                  return (
                    <line key={i} x1={toX(fromNode.lon)} y1={toY(fromNode.lat)} x2={toX(toNode.lon)} y2={toY(toNode.lat)} stroke="#D4D4C8" strokeWidth="0.3" />
                  );
                })}
                {nodes.map((n) => (
                  <circle key={n.id} cx={toX(n.lon)} cy={toY(n.lat)} r="1.4" fill={svgColor[n.district] ?? "#9C9C95"} />
                ))}
              </svg>
            </div>
            <div className="flex gap-5 mt-4 text-sm font-medium">
              {Object.entries(districtColors).map(([name, color]) => (
                <span key={name} className="flex items-center gap-1.5">
                  <span className={`w-3.5 h-3.5 rounded-full ${color}`} />
                  <span className="text-michi-body">{name}</span>
                  <span className="text-michi-muted">({districts[name] ?? 0})</span>
                </span>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>District Breakdown</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(districts).sort((a, b) => b[1] - a[1]).map(([name, count]) => {
                const pct = Math.round((count / stats.total_stations) * 100);
                return (
                  <div key={name}>
                    <div className="flex justify-between text-sm mb-1.5">
                      <span className="font-semibold text-michi-dark">{name}</span>
                      <span className="text-michi-muted font-medium">{count} stations ({pct}%)</span>
                    </div>
                    <div className="h-3 bg-michi-warm rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${districtColors[name] ?? "bg-michi-muted"}`} style={{ width: `${pct}%` }} />
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
          <div className="max-h-72 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-michi-border">
                  <th className="text-left py-2.5 font-semibold text-michi-muted">ID</th>
                  <th className="text-left py-2.5 font-semibold text-michi-muted">Name</th>
                  <th className="text-left py-2.5 font-semibold text-michi-muted">District</th>
                  <th className="text-right py-2.5 font-semibold text-michi-muted">Lat</th>
                  <th className="text-right py-2.5 font-semibold text-michi-muted">Lon</th>
                </tr>
              </thead>
              <tbody>
                {nodes.slice(0, 50).map((n) => (
                  <tr key={n.id} className="border-b border-michi-border/50 hover:bg-michi-warm transition-colors">
                    <td className="py-2.5 font-mono text-xs text-michi-body">{n.id}</td>
                    <td className="py-2.5 font-semibold text-michi-dark">{n.name}</td>
                    <td className="py-2.5">
                      <span className={`px-2.5 py-1 rounded-full text-xs font-semibold text-white ${districtColors[n.district] ?? "bg-michi-muted"}`}>
                        {n.district}
                      </span>
                    </td>
                    <td className="py-2.5 text-right font-mono text-xs text-michi-body">{n.lat.toFixed(4)}</td>
                    <td className="py-2.5 text-right font-mono text-xs text-michi-body">{n.lon.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {nodes.length > 50 && <p className="text-sm text-michi-muted mt-3 text-center font-medium">Showing first 50 of {nodes.length} stations</p>}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}