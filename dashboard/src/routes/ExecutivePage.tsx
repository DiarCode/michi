import { useQuery } from "@tanstack/react-query";
import { fetchExecutiveKPIs, fetchExecutiveTrends, fetchROISummary, fetchFinancialSummary } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  BarChartIcon, Alert01Icon, ArrowUp01Icon, DollarSignIcon, TargetIcon, Clock01Icon,
  UserMultipleIcon, Bus01Icon, ActivityIcon, ArrowUpRightIcon, ArrowDownRightIcon, Fuel01Icon, Wrench01Icon, UserCheckIcon, BanknoteIcon,
} from "@/lib/icons";
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
  if (value > 0) return <span className="flex items-center gap-0.5 text-chart-2 text-sm font-semibold"><HugeiconsIcon icon={ArrowUpRightIcon} size={14} />+{value.toFixed(1)}%</span>;
  if (value < 0) return <span className="flex items-center gap-0.5 text-destructive text-sm font-semibold"><HugeiconsIcon icon={ArrowDownRightIcon} size={14} />{value.toFixed(1)}%</span>;
  return <span className="text-sm text-muted-foreground">—</span>;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-foreground text-background rounded-xl px-3.5 py-2.5 shadow-lg text-xs">
      <p className="text-muted-foreground mb-1">{label}</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} className="font-semibold">{entry.value.toLocaleString()} passengers</p>
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
      <div>
        <h1 className="text-3xl font-extrabold text-foreground">Executive Dashboard</h1>
      </div>

      {loadingKpis ? (
        <GridSkeleton />
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium mb-2">
                  <HugeiconsIcon icon={UserMultipleIcon} size={14} /> Daily Ridership
                </div>
                <p className="text-3xl font-extrabold text-foreground">{dailyRidership.toLocaleString()}</p>
                <TrendIndicator value={(trends?.change_pct as number) ?? 0} />
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium mb-2">
                  <HugeiconsIcon icon={BanknoteIcon} size={14} /> Revenue Today
                </div>
                <p className="text-3xl font-extrabold text-chart-2">₸{formatKZT(revenueToday)}</p>
                <p className="text-sm text-muted-foreground mt-1">Avg fare: ₸{(kpis?.avg_fare_kzt ?? 90) as number}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium mb-2">
                  <HugeiconsIcon icon={Clock01Icon} size={14} /> On-Time Performance
                </div>
                <p className={`text-3xl font-extrabold ${(kpis?.on_time_performance ?? 0) >= 90 ? "text-chart-2" : (kpis?.on_time_performance ?? 0) >= 80 ? "text-chart-4" : "text-destructive"}`}>
                  {kpis?.on_time_performance ?? "—"}%
                </p>
                <p className="text-sm text-muted-foreground mt-1">MAPE: {kpis?.prediction_accuracy_mape ?? "—"}%</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium mb-2">
                  <HugeiconsIcon icon={Alert01Icon} size={14} className="text-destructive" /> Critical Alerts
                </div>
                <p className="text-3xl font-extrabold text-foreground">{kpis?.critical_alerts ?? "—"}</p>
                <p className="text-sm text-muted-foreground mt-1">{kpis?.alerts_today ?? 0} total today</p>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium mb-2">
                  <HugeiconsIcon icon={TargetIcon} size={14} /> Interventions
                </div>
                <p className="text-3xl font-extrabold text-foreground">{kpis?.interventions_today ?? "—"}</p>
                <p className="text-sm text-muted-foreground mt-1">{kpis?.completed_interventions ?? 0} completed</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium mb-2">
                  <HugeiconsIcon icon={Bus01Icon} size={14} /> Fleet Size
                </div>
                <p className="text-3xl font-extrabold text-foreground">{fleetSize}</p>
                <p className="text-sm text-muted-foreground mt-1">{kpis?.active_routes ?? 0} routes</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium mb-2">
                  <HugeiconsIcon icon={BarChartIcon} size={14} /> Operating Ratio
                </div>
                <p className={`text-3xl font-extrabold ${opRatio >= 1 ? "text-chart-2" : "text-destructive"}`}>
                  {opRatio.toFixed(2)}
                </p>
                <p className="text-sm text-muted-foreground mt-1">{opRatio >= 1 ? "Profitable" : "Below break-even"}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium mb-2">
                  <HugeiconsIcon icon={ActivityIcon} size={14} /> Overcrowding Prevented
                </div>
                <p className="text-3xl font-extrabold text-chart-2">{kpis?.overcrowding_prevented ?? "—"}</p>
                <p className="text-sm text-muted-foreground mt-1">incidents this period</p>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <HugeiconsIcon icon={ArrowUp01Icon} size={18} className="text-chart-2" /> 30-Day Ridership Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            {trendData.length === 0 ? (
              <p className="text-muted-foreground text-base py-8 text-center">No trend data available</p>
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={trendData}>
                  <defs>
                    <linearGradient id="riderGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--chart-2)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="var(--chart-2)" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} tickFormatter={(v: string) => v.slice(5)} interval="preserveStartEnd" stroke="var(--border)" />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} domain={[0, maxRidership]} tickFormatter={(v: number) => formatKZT(v)} stroke="var(--border)" />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="ridership" stroke="var(--chart-2)" strokeWidth={2.5} fill="url(#riderGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
            {trends && (
              <div className="flex justify-between mt-3 text-sm text-muted-foreground font-medium">
                <span>Avg: {(trends.avg_daily as number)?.toLocaleString()} passengers/day</span>
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
              <HugeiconsIcon icon={DollarSignIcon} size={18} className="text-chart-2" /> Financial Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            {financialDaily ? (
              <div className="space-y-5">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-chart-2/10 rounded-xl p-4">
                    <p className="text-xs text-muted-foreground font-medium">Revenue</p>
                    <p className="text-xl font-extrabold text-chart-2 mt-1">₸{formatKZT(financialDaily.revenue_kzt)}</p>
                  </div>
                  <div className="bg-destructive/10 rounded-xl p-4">
                    <p className="text-xs text-muted-foreground font-medium">Cost</p>
                    <p className="text-xl font-extrabold text-destructive mt-1">₸{formatKZT(financialDaily.total_cost_kzt)}</p>
                  </div>
                  <div className={`${(financialDaily.net_income_kzt ?? 0) >= 0 ? "bg-chart-2/10" : "bg-destructive/10"} rounded-xl p-4`}>
                    <p className="text-xs text-muted-foreground font-medium">Net</p>
                    <p className={`text-xl font-extrabold ${(financialDaily.net_income_kzt ?? 0) >= 0 ? "text-chart-2" : "text-destructive"} mt-1`}>
                      ₸{formatKZT(financialDaily.net_income_kzt ?? 0)}
                    </p>
                  </div>
                </div>

                {costBreakdown && (
                  <div>
                    <p className="text-sm font-semibold text-foreground mb-3">Cost Breakdown</p>
                    <div className="space-y-2.5">
                      {Object.entries(costBreakdown).map(([key, val]) => {
                        const labels: Record<string, { icon: any; label: string }> = {
                          fleet_operations: { icon: Bus01Icon, label: "Fleet" },
                          fuel: { icon: Fuel01Icon, label: "Fuel" },
                          staff: { icon: UserCheckIcon, label: "Staff" },
                          maintenance: { icon: Wrench01Icon, label: "Maintenance" },
                        };
                        const info = labels[key] ?? { icon: BanknoteIcon, label: key };
                        const IconComp = info.icon;
                        const pct = val / Math.max(financialDaily.total_cost_kzt ?? 1, 1) * 100;
                        return (
                          <div key={key} className="flex items-center gap-2.5">
                            <HugeiconsIcon icon={IconComp} size={14} className="text-muted-foreground flex-shrink-0" />
                            <span className="text-sm text-muted-foreground w-24 font-medium">{info.label}</span>
                            <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                              <div className="h-full bg-destructive/60 rounded-full" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-sm font-mono text-muted-foreground w-20 text-right">₸{formatKZT(val)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {monthlyProjection && (
                  <div className="pt-4 border-t border-border text-sm text-muted-foreground font-medium space-y-1.5">
                    <p className="font-semibold text-foreground">Monthly Projection</p>
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
              <p className="text-muted-foreground text-base py-8 text-center">No financial data available</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HugeiconsIcon icon={TargetIcon} size={18} className="text-chart-2" /> Intervention ROI
          </CardTitle>
        </CardHeader>
        <CardContent>
          {roi ? (
            <div className="grid grid-cols-2 lg:grid-cols-6 gap-5">
              <div className="text-center">
                <p className="text-sm text-muted-foreground font-medium mb-2">Total</p>
                <p className="text-3xl font-extrabold text-foreground">{roi.total_interventions as number}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground font-medium mb-2">Completed</p>
                <p className="text-3xl font-extrabold text-chart-2">{roi.completed as number}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground font-medium mb-2">Benefit</p>
                <p className="text-3xl font-extrabold text-chart-2">${(roi.estimated_benefit_usd as number).toLocaleString()}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground font-medium mb-2">Net ROI</p>
                <p className={`text-3xl font-extrabold ${(roi.net_roi_pct as number) >= 0 ? "text-chart-2" : "text-destructive"}`}>
                  {roi.net_roi_pct as number}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground font-medium mb-2">Ridership Saved</p>
                <p className="text-3xl font-extrabold text-foreground">{(roi.estimated_ridership_saved as number).toLocaleString()}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground font-medium mb-2">Fuel Saved</p>
                <p className="text-3xl font-extrabold text-foreground">{(roi.fuel_savings_liters as number).toLocaleString()} L</p>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground text-base py-8 text-center">No ROI data available</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}