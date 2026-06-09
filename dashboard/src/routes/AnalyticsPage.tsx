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

  if (loadingSummary) return <div className="p-8 space-y-8"><GridSkeleton /><ChartSkeleton /></div>;

  const districts = summary?.ridership_by_district ?? {};
  const routes = summary?.route_performance ?? [];
  const hourly = summary?.hourly_distribution ?? [];
  const maxHourly = Math.max(...hourly.map((h) => h.ridership), 1);

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Analytics</h1>
        <p className="text-base text-michi-muted mt-1">Ridership patterns, district performance, and trend analysis</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-michi-muted font-medium">Total Districts</span>
              <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                <MapPin size={16} className="text-michi-lime-dark" />
              </div>
            </div>
            <p className="text-3xl font-extrabold text-michi-dark">{Object.keys(districts).length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-michi-muted font-medium">Total Ridership</span>
              <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                <Users size={16} className="text-michi-lime-dark" />
              </div>
            </div>
            <p className="text-3xl font-extrabold text-michi-dark">{Object.values(districts).reduce((s, d) => s + d.total, 0).toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-michi-muted font-medium">Avg On-Time</span>
              <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                <Clock size={16} className="text-michi-lime-dark" />
              </div>
            </div>
            <p className="text-3xl font-extrabold text-michi-dark">{routes.length ? Math.round(routes.reduce((s, r) => s + r.on_time_pct, 0) / routes.length) : "—"}%</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-michi-muted font-medium">Trend</span>
              <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                <TrendingUp size={16} className="text-michi-lime-dark" />
              </div>
            </div>
            <p className="text-3xl font-extrabold text-michi-dark capitalize">{trends?.trend ?? "—"}</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>District Ridership</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(districts).map(([name, data]) => {
                const maxTotal = Math.max(...Object.values(districts).map((d) => d.total), 1);
                const pct = Math.round((data.total / maxTotal) * 100);
                return (
                  <div key={name}>
                    <div className="flex justify-between text-sm mb-1.5">
                      <span className="font-semibold text-michi-dark">{name}</span>
                      <span className="text-michi-muted font-medium">{data.total.toLocaleString()} · peak {data.peak_hour}:00</span>
                    </div>
                    <div className="h-3 bg-michi-warm rounded-full overflow-hidden">
                      <div className="h-full bg-michi-lime rounded-full transition-all" style={{ width: `${pct}%` }} />
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
                <tr className="border-b border-michi-border">
                  <th className="text-left py-2.5 font-semibold text-michi-muted">Route</th>
                  <th className="text-right py-2.5 font-semibold text-michi-muted">On-Time</th>
                  <th className="text-right py-2.5 font-semibold text-michi-muted">Avg Wait</th>
                  <th className="text-right py-2.5 font-semibold text-michi-muted">Daily Riders</th>
                </tr>
              </thead>
              <tbody>
                {routes.map((r) => (
                  <tr key={r.route_id} className="border-b border-michi-border/50 hover:bg-michi-warm transition-colors">
                    <td className="py-2.5 font-semibold text-michi-dark">{r.name}</td>
                    <td className={`text-right py-2.5 font-mono font-semibold ${r.on_time_pct >= 90 ? "text-michi-lime-dark" : r.on_time_pct >= 85 ? "text-michi-amber" : "text-michi-red"}`}>
                      {r.on_time_pct}%
                    </td>
                    <td className="text-right py-2.5 font-mono text-michi-body">{r.avg_wait_min}min</td>
                    <td className="text-right py-2.5 font-mono text-michi-body">{r.daily_ridership.toLocaleString()}</td>
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
          <span className="text-sm text-michi-muted font-medium">Peak ridership by hour</span>
        </CardHeader>
        <CardContent>
          <div className="flex items-end gap-1.5 h-44">
            {hourly.map((h) => {
              const pct = (h.ridership / maxHourly) * 100;
              return (
                <div key={h.hour} className="flex-1 flex flex-col items-center gap-1">
                  <div className="w-full bg-michi-lime rounded-t-lg transition-all" style={{ height: `${pct}%`, minHeight: "3px" }} title={`${h.hour}:00 — ${h.ridership.toLocaleString()}`} />
                  {h.hour % 3 === 0 && <span className="text-xs text-michi-muted font-medium">{h.hour}</span>}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>Ridership Trends</CardTitle>
          <div className="flex gap-2">
            {[7, 30, 90, 365].map((d) => (
              <button
                key={d}
                onClick={() => setDays(d)}
                className={`px-3.5 py-1.5 text-xs rounded-full font-semibold transition-all ${
                  days === d
                    ? "bg-michi-dark text-white shadow-sm"
                    : "bg-white border border-michi-border text-michi-body hover:bg-michi-warm"
                }`}
              >
                {d}d
              </button>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          {trends && (
            <div className="space-y-3">
              <div className="flex justify-between text-sm text-michi-muted font-medium">
                <span>Avg daily: {trends.avg_daily.toLocaleString()}</span>
                <span className={trends.change_pct >= 0 ? "text-michi-lime-dark" : "text-michi-red"}>
                  {trends.change_pct >= 0 ? "+" : ""}{trends.change_pct}% change
                </span>
              </div>
              <div className="flex items-end gap-0.5 h-36">
                {trends.trends.slice(-Math.min(60, trends.trends.length)).map((t, i) => {
                  const maxVal = Math.max(...trends.trends.map((x) => x.ridership), 1);
                  const pct = (t.ridership / maxVal) * 100;
                  return (
                    <div key={i} className="flex-1 bg-michi-lime rounded-t" style={{ height: `${pct}%`, minHeight: "2px" }} title={`${t.date}: ${t.ridership.toLocaleString()}`} />
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