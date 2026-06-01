import { useQuery } from "@tanstack/react-query";
import { fetchExecutiveKPIs, fetchExecutiveTrends, fetchROISummary, fetchFinancialSummary } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BarChart3, AlertTriangle, TrendingUp, DollarSign, Target, Clock,
  Users, Bus, Activity, ArrowUpRight, ArrowDownRight, Fuel, Wrench, UserCheck, Banknote,
} from "lucide-react";
import { GridSkeleton } from "@/components/ui/skeleton";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

function formatKZT(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return value.toLocaleString();
}

function TrendIndicator({ value }: { value: number }) {
  if (value > 0) return <span className="flex items-center gap-0.5 text-green-600 dark:text-green-400 text-xs"><ArrowUpRight className="h-3 w-3" />+{value.toFixed(1)}%</span>;
  if (value < 0) return <span className="flex items-center gap-0.5 text-red-600 dark:text-red-400 text-xs"><ArrowDownRight className="h-3 w-3" />{value.toFixed(1)}%</span>;
  return <span className="text-xs text-gray-400">—</span>;
}

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
  const { data: financial } = useQuery({
    queryKey: ["executive-financial"],
    queryFn: fetchFinancialSummary,
    staleTime: 300000,
  });

  const trendData = trends?.trends ?? [];
  const maxRidership = Math.max(...trendData.map((t) => t.ridership), 1);

  const dailyRidership = (kpis?.daily_ridership ?? 0) as number;
  const revenueToday = (kpis?.revenue_today_kzt ?? 0) as number;
  const opRatio = (kpis?.operating_ratio ?? 0) as number;
  const fleetSize = (kpis?.fleet_size ?? 0) as number;

  const financialDaily = financial?.daily as Record<string, number> | undefined;
  const costBreakdown = financial?.cost_breakdown as Record<string, number> | undefined;
  const monthlyProjection = financial?.monthly_projection as Record<string, number> | undefined;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold dark:text-white">Executive Dashboard</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Strategic KPIs, financial metrics, trends, and ROI</p>
        </div>
        <Badge variant="default" className="text-xs">
          {new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" })}
        </Badge>
      </div>

      {loadingKpis ? (
        <GridSkeleton />
      ) : (
        <>
          {/* Core KPIs */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <Users className="h-3 w-3" /> Daily Ridership
                </div>
                <p className="text-2xl font-bold dark:text-white">{dailyRidership.toLocaleString()}</p>
                <TrendIndicator value={(trends?.change_pct as number) ?? 0} />
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <Banknote className="h-3 w-3" /> Revenue Today
                </div>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">₸{formatKZT(revenueToday)}</p>
                <p className="text-xs text-gray-400">Avg fare: ₸{(kpis?.avg_fare_kzt ?? 90) as number}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <Clock className="h-3 w-3" /> On-Time Performance
                </div>
                <p className={`text-2xl font-bold ${(kpis?.on_time_performance ?? 0) >= 90 ? "text-green-600 dark:text-green-400" : (kpis?.on_time_performance ?? 0) >= 80 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400"}`}>
                  {kpis?.on_time_performance ?? "—"}%
                </p>
                <p className="text-xs text-gray-400">MAPE: {kpis?.prediction_accuracy_mape ?? "—"}%</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <AlertTriangle className="h-3 w-3 text-red-500" /> Critical Alerts
                </div>
                <p className="text-2xl font-bold dark:text-white">{kpis?.critical_alerts ?? "—"}</p>
                <p className="text-xs text-gray-400">{kpis?.alerts_today ?? 0} total today</p>
              </CardContent>
            </Card>
          </div>

          {/* Operations KPIs */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <Target className="h-3 w-3 text-blue-500" /> Interventions
                </div>
                <p className="text-2xl font-bold dark:text-white">{kpis?.interventions_today ?? "—"}</p>
                <p className="text-xs text-gray-400">{kpis?.completed_interventions ?? 0} completed</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <Bus className="h-3 w-3" /> Fleet Size
                </div>
                <p className="text-2xl font-bold dark:text-white">{fleetSize}</p>
                <p className="text-xs text-gray-400">{kpis?.active_routes ?? 0} routes</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <BarChart3 className="h-3 w-3" /> Operating Ratio
                </div>
                <p className={`text-2xl font-bold ${opRatio >= 1 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
                  {opRatio.toFixed(2)}
                </p>
                <p className="text-xs text-gray-400">{opRatio >= 1 ? "Profitable" : "Below break-even"}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <Activity className="h-3 w-3" /> Overcrowding Prevented
                </div>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">{kpis?.overcrowding_prevented ?? "—"}</p>
                <p className="text-xs text-gray-400">incidents this period</p>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* 30-Day Ridership Trend */}
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
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={trendData}>
                  <defs>
                    <linearGradient id="riderGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="date" tick={{ fontSize: 9 }} tickFormatter={(v: string) => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9 }} domain={[0, maxRidership]} tickFormatter={(v: number) => formatKZT(v)} />
                  <Tooltip
                    contentStyle={{ fontSize: 11, borderRadius: 8 }}
                    formatter={(value: number) => [`${value.toLocaleString()} pax`, "Ridership"]}
                    labelFormatter={(label: string) => label}
                  />
                  <Area type="monotone" dataKey="ridership" stroke="#3b82f6" strokeWidth={2} fill="url(#riderGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
            {trends && (
              <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
                <span>Avg: {(trends.avg_daily as number)?.toLocaleString()} pax/day</span>
                <span className="flex items-center gap-1">
                  Trend: {trends.trend as string}
                  <TrendIndicator value={trends.change_pct as number} />
                </span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Financial Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <DollarSign className="h-4 w-4" /> Financial Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            {financialDaily ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                    <p className="text-[10px] text-gray-500 dark:text-gray-400">Revenue</p>
                    <p className="text-lg font-bold text-green-600 dark:text-green-400">₸{formatKZT(financialDaily.revenue_kzt)}</p>
                  </div>
                  <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
                    <p className="text-[10px] text-gray-500 dark:text-gray-400">Cost</p>
                    <p className="text-lg font-bold text-red-600 dark:text-red-400">₸{formatKZT(financialDaily.total_cost_kzt)}</p>
                  </div>
                  <div className={`${(financialDaily.net_income_kzt ?? 0) >= 0 ? "bg-green-50 dark:bg-green-900/20" : "bg-red-50 dark:bg-red-900/20"} rounded-lg p-3`}>
                    <p className="text-[10px] text-gray-500 dark:text-gray-400">Net</p>
                    <p className={`text-lg font-bold ${(financialDaily.net_income_kzt ?? 0) >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
                      ₸{formatKZT(financialDaily.net_income_kzt ?? 0)}
                    </p>
                  </div>
                </div>

                {/* Cost breakdown */}
                {costBreakdown && (
                  <div>
                    <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">Cost Breakdown</p>
                    <div className="space-y-2">
                      {Object.entries(costBreakdown).map(([key, val]) => {
                        const labels: Record<string, { icon: typeof Bus; label: string }> = {
                          fleet_operations: { icon: Bus, label: "Fleet" },
                          fuel: { icon: Fuel, label: "Fuel" },
                          staff: { icon: UserCheck, label: "Staff" },
                          maintenance: { icon: Wrench, label: "Maintenance" },
                        };
                        const info = labels[key] ?? { icon: Banknote, label: key };
                        const IconComp = info.icon;
                        const pct = val / Math.max(financialDaily.total_cost_kzt ?? 1, 1) * 100;
                        return (
                          <div key={key} className="flex items-center gap-2">
                            <IconComp className="h-3 w-3 text-gray-400 flex-shrink-0" />
                            <span className="text-xs text-gray-600 dark:text-gray-400 w-24">{info.label}</span>
                            <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                              <div className="h-full bg-red-400 dark:bg-red-500 rounded-full" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-xs font-mono text-gray-500 w-16 text-right">₸{formatKZT(val)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Monthly projection */}
                {monthlyProjection && (
                  <div className="pt-3 border-t dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400 space-y-1">
                    <p className="font-medium">Monthly Projection</p>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      <span>Revenue: ₸{formatKZT(monthlyProjection.revenue_kzt ?? 0)}</span>
                      <span>Cost: ₸{formatKZT(monthlyProjection.total_cost_kzt ?? 0)}</span>
                      <span>Net: ₸{formatKZT(monthlyProjection.net_income_kzt ?? 0)}</span>
                      <span>Cost/Pax: ₸{(financial?.cost_per_passenger_kzt ?? 0) as number}</span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-gray-400 text-sm">No financial data available</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ROI Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Target className="h-4 w-4" /> Intervention ROI
          </CardTitle>
        </CardHeader>
        <CardContent>
          {roi ? (
            <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
              <div className="text-center">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Total</p>
                <p className="text-xl font-bold dark:text-white">{roi.total_interventions as number}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Completed</p>
                <p className="text-xl font-bold text-green-600 dark:text-green-400">{roi.completed as number}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Benefit</p>
                <p className="text-xl font-bold text-green-600 dark:text-green-400">${(roi.estimated_benefit_usd as number).toLocaleString()}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Net ROI</p>
                <p className={`text-xl font-bold ${(roi.net_roi_pct as number) >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
                  {roi.net_roi_pct as number}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Ridership Saved</p>
                <p className="text-xl font-bold dark:text-white">{(roi.estimated_ridership_saved as number).toLocaleString()}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Fuel Saved</p>
                <p className="text-xl font-bold dark:text-white">{(roi.fuel_savings_liters as number).toLocaleString()} L</p>
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No ROI data available</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}