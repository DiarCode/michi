import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchAnalyticsSummary, fetchAnalyticsTrends } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { TrendingUp, Users, Clock, MapPin } from "lucide-react";
import { GridSkeleton, ChartSkeleton } from "@/components/ui/skeleton";

export default function AnalyticsPage() {
  const [days, setDays] = useState(30);
  const { data: summary, isLoading: loadingSummary } = useQuery({ queryKey: ["analytics-summary"], queryFn: fetchAnalyticsSummary });
  const { data: trends } = useQuery({ queryKey: ["analytics-trends", days], queryFn: () => fetchAnalyticsTrends(days) });

  if (loadingSummary) return <div className="p-6 space-y-6"><GridSkeleton /><ChartSkeleton /></div>;

  const districts = summary?.ridership_by_district ?? {};
  const routes = summary?.route_performance ?? [];
  const hourly = summary?.hourly_distribution ?? [];
  const maxHourly = Math.max(...hourly.map((h) => h.ridership), 1);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Analytics</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400">Ridership analytics, district breakdown, and performance metrics.</p>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Total Districts</CardTitle>
            <MapPin className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{Object.keys(districts).length}</p></CardContent>
        </Card>
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Total Ridership</CardTitle>
            <Users className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{Object.values(districts).reduce((s, d) => s + d.total, 0).toLocaleString()}</p></CardContent>
        </Card>
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Avg On-Time</CardTitle>
            <Clock className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{routes.length ? Math.round(routes.reduce((s, r) => s + r.on_time_pct, 0) / routes.length) : "—"}%</p></CardContent>
        </Card>
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Trend</CardTitle>
            <TrendingUp className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent><p className="text-2xl font-bold capitalize">{trends?.trend ?? "—"}</p></CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>District Ridership</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(districts).map(([name, data]) => {
                const maxTotal = Math.max(...Object.values(districts).map((d) => d.total), 1);
                const pct = Math.round((data.total / maxTotal) * 100);
                return (
                  <div key={name}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium">{name}</span>
                      <span className="text-gray-500 dark:text-gray-400">{data.total.toLocaleString()} · peak {data.peak_hour}:00</span>
                    </div>
                    <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500 rounded-full" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Route Performance</CardTitle></CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Route</th>
                  <th className="text-right py-2">On-Time</th>
                  <th className="text-right py-2">Avg Wait</th>
                  <th className="text-right py-2">Daily Riders</th>
                </tr>
              </thead>
              <tbody>
                {routes.map((r) => (
                  <tr key={r.route_id} className="border-b hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="py-2 font-medium">{r.name}</td>
                    <td className={`text-right py-2 font-mono ${r.on_time_pct >= 90 ? "text-green-600" : r.on_time_pct >= 85 ? "text-amber-600" : "text-red-600"}`}>
                      {r.on_time_pct}%
                    </td>
                    <td className="text-right py-2 font-mono">{r.avg_wait_min}min</td>
                    <td className="text-right py-2 font-mono">{r.daily_ridership.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>Hourly Distribution</CardTitle>
          <span className="text-xs text-gray-500 dark:text-gray-400">Peak ridership by hour</span>
        </CardHeader>
        <CardContent>
          <div className="flex items-end gap-1 h-40">
            {hourly.map((h) => {
              const pct = (h.ridership / maxHourly) * 100;
              return (
                <div key={h.hour} className="flex-1 flex flex-col items-center gap-1">
                  <div className="w-full bg-blue-500 rounded-t transition-all" style={{ height: `${pct}%`, minHeight: "2px" }} title={`${h.hour}:00 — ${h.ridership.toLocaleString()}`} />
                  {h.hour % 3 === 0 && <span className="text-[9px] text-gray-400">{h.hour}</span>}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>Ridership Trends</CardTitle>
          <div className="flex items-center gap-2">
            <select className="text-sm border rounded px-2 py-1 dark:bg-gray-800" value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {[7, 30, 90, 365].map((d) => <option key={d} value={d}>{d}d</option>)}
            </select>
          </div>
        </CardHeader>
        <CardContent>
          {trends && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400">
                <span>Avg daily: {trends.avg_daily.toLocaleString()}</span>
                <span className={trends.change_pct >= 0 ? "text-green-600" : "text-red-600"}>
                  {trends.change_pct >= 0 ? "+" : ""}{trends.change_pct}% change
                </span>
              </div>
              <div className="flex items-end gap-0.5 h-32">
                {trends.trends.slice(-Math.min(60, trends.trends.length)).map((t, i) => {
                  const maxVal = Math.max(...trends.trends.map((x) => x.ridership), 1);
                  const pct = (t.ridership / maxVal) * 100;
                  return (
                    <div key={i} className="flex-1 bg-emerald-500 rounded-t" style={{ height: `${pct}%`, minHeight: "1px" }} title={`${t.date}: ${t.ridership.toLocaleString()}`} />
                  );
                })}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}