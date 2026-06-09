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
  if (value > 0) return <span className="flex items-center gap-0.5 text-michi-lime-dark text-sm font-semibold"><ArrowUpRight size={14} />+{value.toFixed(1)}%</span>;
  if (value < 0) return <span className="flex items-center gap-0.5 text-michi-red text-sm font-semibold"><ArrowDownRight size={14} />{value.toFixed(1)}%</span>;
  return <span className="text-sm text-michi-muted">—</span>;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-michi-dark text-white rounded-xl px-3.5 py-2.5 shadow-tooltip text-xs">
      <p className="text-michi-muted mb-1">{label}</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} className="font-semibold">{entry.value.toLocaleString()} pax</p>
      ))}
    </div>
  );
};

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
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold text-michi-dark">Executive Dashboard</h1>
          <p className="text-base text-michi-muted mt-1">Strategic KPIs, financial metrics, trends, and ROI analysis</p>
        </div>
        <Badge variant="default" className="text-sm font-semibold">
          {new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" })}
        </Badge>
      </div>

      {loadingKpis ? (
        <GridSkeleton />
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <Users size={14} /> Daily Ridership
                </div>
                <p className="text-3xl font-extrabold text-michi-dark">{dailyRidership.toLocaleString()}</p>
                <TrendIndicator value={(trends?.change_pct as number) ?? 0} />
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <Banknote size={14} /> Revenue Today
                </div>
                <p className="text-3xl font-extrabold text-michi-lime-dark">₸{formatKZT(revenueToday)}</p>
                <p className="text-sm text-michi-muted mt-1">Avg fare: ₸{(kpis?.avg_fare_kzt ?? 90) as number}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <Clock size={14} /> On-Time Performance
                </div>
                <p className={`text-3xl font-extrabold ${(kpis?.on_time_performance ?? 0) >= 90 ? "text-michi-lime-dark" : (kpis?.on_time_performance ?? 0) >= 80 ? "text-michi-amber" : "text-michi-red"}`}>
                  {kpis?.on_time_performance ?? "—"}%
                </p>
                <p className="text-sm text-michi-muted mt-1">MAPE: {kpis?.prediction_accuracy_mape ?? "—"}%</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <AlertTriangle size={14} className="text-michi-red" /> Critical Alerts
                </div>
                <p className="text-3xl font-extrabold text-michi-dark">{kpis?.critical_alerts ?? "—"}</p>
                <p className="text-sm text-michi-muted mt-1">{kpis?.alerts_today ?? 0} total today</p>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <Target size={14} /> Interventions
                </div>
                <p className="text-3xl font-extrabold text-michi-dark">{kpis?.interventions_today ?? "—"}</p>
                <p className="text-sm text-michi-muted mt-1">{kpis?.completed_interventions ?? 0} completed</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <Bus size={14} /> Fleet Size
                </div>
                <p className="text-3xl font-extrabold text-michi-dark">{fleetSize}</p>
                <p className="text-sm text-michi-muted mt-1">{kpis?.active_routes ?? 0} routes</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <BarChart3 size={14} /> Operating Ratio
                </div>
                <p className={`text-3xl font-extrabold ${opRatio >= 1 ? "text-michi-lime-dark" : "text-michi-red"}`}>
                  {opRatio.toFixed(2)}
                </p>
                <p className="text-sm text-michi-muted mt-1">{opRatio >= 1 ? "Profitable" : "Below break-even"}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                  <Activity size={14} /> Overcrowding Prevented
                </div>
                <p className="text-3xl font-extrabold text-michi-lime-dark">{kpis?.overcrowding_prevented ?? "—"}</p>
                <p className="text-sm text-michi-muted mt-1">incidents this period</p>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp size={18} className="text-michi-lime-dark" /> 30-Day Ridership Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            {trendData.length === 0 ? (
              <p className="text-michi-muted text-base py-8 text-center">No trend data available</p>
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={trendData}>
                  <defs>
                    <linearGradient id="riderGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#B1E743" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#B1E743" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E8E8E0" />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#9C9C95' }} tickFormatter={(v: string) => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 10, fill: '#9C9C95' }} domain={[0, maxRidership]} tickFormatter={(v: number) => formatKZT(v)} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="ridership" stroke="#B1E743" strokeWidth={2.5} fill="url(#riderGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
            {trends && (
              <div className="flex justify-between mt-3 text-sm text-michi-muted font-medium">
                <span>Avg: {(trends.avg_daily as number)?.toLocaleString()} pax/day</span>
                <span className="flex items-center gap-1.5">
                  Trend: {trends.trend as string}
                  <TrendIndicator value={trends.change_pct as number} />
                </span>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSign size={18} className="text-michi-lime-dark" /> Financial Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            {financialDaily ? (
              <div className="space-y-5">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-michi-lime/10 rounded-xl p-4">
                    <p className="text-xs text-michi-muted font-medium">Revenue</p>
                    <p className="text-xl font-extrabold text-michi-lime-dark mt-1">₸{formatKZT(financialDaily.revenue_kzt)}</p>
                  </div>
                  <div className="bg-michi-red/8 rounded-xl p-4">
                    <p className="text-xs text-michi-muted font-medium">Cost</p>
                    <p className="text-xl font-extrabold text-michi-red mt-1">₸{formatKZT(financialDaily.total_cost_kzt)}</p>
                  </div>
                  <div className={`${(financialDaily.net_income_kzt ?? 0) >= 0 ? "bg-michi-lime/10" : "bg-michi-red/8"} rounded-xl p-4`}>
                    <p className="text-xs text-michi-muted font-medium">Net</p>
                    <p className={`text-xl font-extrabold ${(financialDaily.net_income_kzt ?? 0) >= 0 ? "text-michi-lime-dark" : "text-michi-red"} mt-1`}>
                      ₸{formatKZT(financialDaily.net_income_kzt ?? 0)}
                    </p>
                  </div>
                </div>

                {costBreakdown && (
                  <div>
                    <p className="text-sm font-semibold text-michi-dark mb-3">Cost Breakdown</p>
                    <div className="space-y-2.5">
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
                          <div key={key} className="flex items-center gap-2.5">
                            <IconComp size={14} className="text-michi-muted flex-shrink-0" />
                            <span className="text-sm text-michi-body w-24 font-medium">{info.label}</span>
                            <div className="flex-1 h-3 bg-michi-warm rounded-full overflow-hidden">
                              <div className="h-full bg-michi-red/60 rounded-full" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-sm font-mono text-michi-muted w-20 text-right">₸{formatKZT(val)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {monthlyProjection && (
                  <div className="pt-4 border-t border-michi-border text-sm text-michi-muted font-medium space-y-1.5">
                    <p className="font-semibold text-michi-dark">Monthly Projection</p>
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
              <p className="text-michi-muted text-base py-8 text-center">No financial data available</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target size={18} className="text-michi-lime-dark" /> Intervention ROI
          </CardTitle>
        </CardHeader>
        <CardContent>
          {roi ? (
            <div className="grid grid-cols-2 lg:grid-cols-6 gap-5">
              <div className="text-center">
                <p className="text-sm text-michi-muted font-medium mb-2">Total</p>
                <p className="text-3xl font-extrabold text-michi-dark">{roi.total_interventions as number}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-michi-muted font-medium mb-2">Completed</p>
                <p className="text-3xl font-extrabold text-michi-lime-dark">{roi.completed as number}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-michi-muted font-medium mb-2">Benefit</p>
                <p className="text-3xl font-extrabold text-michi-lime-dark">${(roi.estimated_benefit_usd as number).toLocaleString()}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-michi-muted font-medium mb-2">Net ROI</p>
                <p className={`text-3xl font-extrabold ${(roi.net_roi_pct as number) >= 0 ? "text-michi-lime-dark" : "text-michi-red"}`}>
                  {roi.net_roi_pct as number}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-michi-muted font-medium mb-2">Ridership Saved</p>
                <p className="text-3xl font-extrabold text-michi-dark">{(roi.estimated_ridership_saved as number).toLocaleString()}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-michi-muted font-medium mb-2">Fuel Saved</p>
                <p className="text-3xl font-extrabold text-michi-dark">{(roi.fuel_savings_liters as number).toLocaleString()} L</p>
              </div>
            </div>
          ) : (
            <p className="text-michi-muted text-base py-8 text-center">No ROI data available</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}