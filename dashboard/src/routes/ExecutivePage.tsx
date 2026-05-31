import { useQuery } from "@tanstack/react-query";
import { fetchExecutiveKPIs, fetchExecutiveTrends, fetchROISummary } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { BarChart3, AlertTriangle, TrendingUp, DollarSign, Target, Clock } from "lucide-react";
import { GridSkeleton } from "@/components/ui/skeleton";

export default function ExecutivePage() {
  const { data: kpis, isLoading: loadingKpis } = useQuery({
    queryKey: ["executive-kpis"],
    queryFn: fetchExecutiveKPIs,
    refetchInterval: 30000,
  });
  const { data: trends } = useQuery({
    queryKey: ["executive-trends"],
    queryFn: () => fetchExecutiveTrends(30),
    staleTime: 300000,
  });
  const { data: roi } = useQuery({
    queryKey: ["executive-roi"],
    queryFn: fetchROISummary,
    staleTime: 300000,
  });

  const trendData = trends?.trends ?? [];
  const maxRidership = Math.max(...trendData.map((t) => t.ridership), 1);

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold dark:text-white">Executive Dashboard</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">Strategic KPIs, trends, and ROI</p>
      </div>

      {loadingKpis ? (
        <GridSkeleton />
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">On-Time</CardTitle>
              <Clock className="h-4 w-4 text-gray-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold dark:text-white">{kpis?.on_time_performance ?? "—"}%</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Critical Alerts</CardTitle>
              <AlertTriangle className="h-4 w-4 text-red-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold dark:text-white">{kpis?.critical_alerts ?? "—"}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Interventions</CardTitle>
              <Target className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold dark:text-white">{kpis?.interventions_today ?? "—"}</div>
              <div className="text-xs text-gray-400">{kpis?.completed_interventions ?? 0} completed</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-xs font-medium text-gray-500 dark:text-gray-400">Prediction MAPE</CardTitle>
              <BarChart3 className="h-4 w-4 text-gray-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold dark:text-white">{kpis?.prediction_accuracy_mape ? `${kpis.prediction_accuracy_mape}%` : "—"}</div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingUp className="h-4 w-4" /> 30-Day Ridership Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            {trendData.length === 0 ? (
              <p className="text-gray-400 text-sm">No trend data available</p>
            ) : (
              <div className="flex items-end gap-px h-32">
                {trendData.map((d, i) => {
                  const pct = (d.ridership / maxRidership) * 100;
                  return (
                    <div key={i} className="flex-1 flex flex-col justify-end">
                      <div className="bg-blue-500 rounded-t-sm" style={{ height: `${pct}%`, minHeight: 2 }} />
                    </div>
                  );
                })}
              </div>
            )}
            <div className="flex justify-between text-[10px] text-gray-400 mt-1">
              <span>30 days ago</span>
              <span>Today</span>
            </div>
            {trends && (
              <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
                <span>Avg: {trends.avg_daily?.toLocaleString()}</span>
                <span>Trend: {trends.trend} ({trends.change_pct > 0 ? "+" : ""}{trends.change_pct}%)</span>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <DollarSign className="h-4 w-4" /> ROI Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            {roi ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Total Interventions</div>
                    <div className="text-lg font-bold dark:text-white">{roi.total_interventions as number}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Completed</div>
                    <div className="text-lg font-bold text-green-600">{roi.completed as number}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Est. Benefit</div>
                    <div className="text-lg font-bold text-green-600">${(roi.estimated_benefit_usd as number).toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Net ROI</div>
                    <div className="text-lg font-bold dark:text-white">{roi.net_roi_pct as number}%</div>
                  </div>
                </div>
                <div className="pt-2 border-t dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
                  <div>Ridership saved: {(roi.estimated_ridership_saved as number).toLocaleString()} pax</div>
                  <div>Wait time saved: {(roi.estimated_wait_time_saved_minutes as number).toLocaleString()} min</div>
                  <div>Fuel saved: {(roi.fuel_savings_liters as number).toLocaleString()} L</div>
                </div>
              </div>
            ) : (
              <p className="text-gray-400 text-sm">No ROI data available</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}