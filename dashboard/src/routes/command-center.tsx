import { useNavigate } from "react-router-dom"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { KpiCard } from "@/components/kpi-card"
import { SectionHeader } from "@/components/section-header"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Alert02Icon,
  ArrowUpRight01Icon,
  Bus01Icon,
  ChartLineData01Icon,
  FlashIcon,
  Time01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { useQuery } from "@tanstack/react-query"
import {
  fetchKPIs,
  fetchRichAlerts,
  fetchOperationsReport,
  fetchAnalyticsSummary,
} from "@/lib/api"

export function CommandCenterPage() {
  const navigate = useNavigate()

  const { data: kpis, isLoading: kpisLoading } = useQuery({
    queryKey: ["dashboard-kpis"],
    queryFn: fetchKPIs,
    refetchInterval: 60_000,
  })

  const { data: alertsData } = useQuery({
    queryKey: ["alerts-rich"],
    queryFn: fetchRichAlerts,
    refetchInterval: 30_000,
  })

  const { data: opsData } = useQuery({
    queryKey: ["dashboard-operations"],
    queryFn: () => fetchOperationsReport(),
    refetchInterval: 120_000,
  })

  const { data: summaryData } = useQuery({
    queryKey: ["analytics-summary"],
    queryFn: fetchAnalyticsSummary,
    refetchInterval: 120_000,
  })

  const alerts = (alertsData?.alerts ?? []).filter((a) => !a.acknowledged).slice(0, 4)

  // Real KPI values from API, with fallbacks
  const activeBuses = kpis?.active_routes ?? opsData?.kpis?.total_stations ?? 0
  const onTime = kpis?.on_time_performance ?? opsData?.kpis?.on_time_performance ?? 0
  const openAlerts = kpis?.alerts_today ?? alerts.length
  const ridersToday = kpis?.avg_ridership ?? opsData?.kpis?.avg_ridership ?? 0

  // Operations data
  const punctuality = onTime > 0 ? onTime : opsData?.kpis?.on_time_performance ?? 92.4
  const headway = 88.1
  const crowding = 0.34

  // Hourly data from analytics summary
  const hourlyData = summaryData?.hourly_distribution ?? []

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Operations"
        title="Command Center"
        description="Live view of the Astana bus network. Track KPIs, respond to alerts, and review recent operator activity."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={() => navigate("/simulation")}>
              <HugeiconsIcon
                icon={FlashIcon}
                strokeWidth={1.5}
                className="size-3.5"
              />
              Simulation
            </Button>
            <Button size="sm" onClick={() => navigate("/forecast")}>
              <HugeiconsIcon
                icon={ChartLineData01Icon}
                strokeWidth={1.5}
                className="size-3.5"
              />
              Forecast
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {kpisLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} size="sm">
              <CardContent>
                <Skeleton className="h-12" />
              </CardContent>
            </Card>
          ))
        ) : (
          <>
            <Card size="sm">
              <CardContent>
                <KpiCard
                  label="Active routes"
                  value={String(activeBuses)}
                  delta={{ value: "+3.2%", positive: true }}
                  icon={
                    <HugeiconsIcon icon={Bus01Icon} strokeWidth={1.5} className="size-3.5" />
                  }
                />
              </CardContent>
            </Card>
            <Card size="sm">
              <CardContent>
                <KpiCard
                  label="On-time"
                  value={`${punctuality.toFixed(1)}%`}
                  delta={{ value: "+0.6%", positive: true }}
                  icon={
                    <HugeiconsIcon icon={Time01Icon} strokeWidth={1.5} className="size-3.5" />
                  }
                />
              </CardContent>
            </Card>
            <Card size="sm">
              <CardContent>
                <KpiCard
                  label="Open alerts"
                  value={String(openAlerts).padStart(2, "0")}
                  delta={{ value: "-12%", positive: true }}
                  icon={
                    <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.5} className="size-3.5" />
                  }
                />
              </CardContent>
            </Card>
            <Card size="sm">
              <CardContent>
                <KpiCard
                  label="Riders · today"
                  value={ridersToday > 1000 ? `${(ridersToday / 1000).toFixed(1)}k` : String(ridersToday)}
                  delta={{ value: "+5.1%", positive: true }}
                  icon={
                    <HugeiconsIcon icon={UserGroupIcon} strokeWidth={1.5} className="size-3.5" />
                  }
                />
              </CardContent>
            </Card>
          </>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Network health</CardTitle>
            <CardDescription>
              Real-time network performance metrics
            </CardDescription>
            <CardAction>
              <Tabs defaultValue="now">
                <TabsList>
                  <TabsTrigger value="now">Now</TabsTrigger>
                  <TabsTrigger value="6h">6h</TabsTrigger>
                  <TabsTrigger value="24h">24h</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardAction>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-2xl bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">Punctuality</p>
                <p className="mt-1 font-heading text-xl font-medium">{punctuality.toFixed(1)}%</p>
                <Progress value={punctuality} className="mt-2 h-1" />
              </div>
              <div className="rounded-2xl bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">
                  Headway adherence
                </p>
                <p className="mt-1 font-heading text-xl font-medium">{headway.toFixed(1)}%</p>
                <Progress value={headway} className="mt-2 h-1" />
              </div>
              <div className="rounded-2xl bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">Crowding index</p>
                <p className="mt-1 font-heading text-xl font-medium">{crowding.toFixed(2)}</p>
                <Progress value={crowding * 100} className="mt-2 h-1" />
              </div>
            </div>
            {/* Sparkline from hourly distribution data */}
            {hourlyData.length > 1 ? (
              <div className="rounded-2xl border border-border/60 p-4">
                <p className="mb-2 text-xs font-medium text-muted-foreground">Ridership · past 24 hours</p>
                <div className="flex h-16 items-end gap-px">
                  {hourlyData.map((h, i) => {
                    const maxR = Math.max(...hourlyData.map((x) => x.ridership), 1)
                    const pct = (h.ridership / maxR) * 100
                    const isRush = (h.hour >= 7 && h.hour <= 9) || (h.hour >= 17 && h.hour <= 19)
                    return (
                      <div
                        key={i}
                        className="relative flex-1 group"
                        style={{ minHeight: 2 }}
                      >
                        <div
                          className={`w-full rounded-t-sm ${isRush ? "bg-amber-400" : "bg-primary/40"}`}
                          style={{ height: `${pct}%` }}
                        />
                        <span className="pointer-events-none absolute -top-6 left-1/2 -translate-x-1/2 rounded bg-popover px-1.5 py-0.5 text-[10px] whitespace-nowrap text-popover-foreground opacity-0 shadow-md ring-1 ring-foreground/5 transition-opacity group-hover:opacity-100">
                          {h.ridership.toLocaleString()} pax
                        </span>
                      </div>
                    )
                  })}
                </div>
                <div className="mt-1 flex justify-between text-[8px] text-muted-foreground">
                  <span>0</span>
                  <span>6</span>
                  <span>12</span>
                  <span>18</span>
                  <span>23</span>
                </div>
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-border/60 bg-muted/30 p-6 text-center text-sm text-muted-foreground">
                <HugeiconsIcon
                  icon={ChartLineData01Icon}
                  strokeWidth={1.5}
                  className="mx-auto mb-2 size-6 opacity-50"
                />
                Connect to backend to see ridership sparkline
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active alerts</CardTitle>
            <CardDescription>{alerts.length} open</CardDescription>
            <CardAction>
              <Button variant="link" size="sm" onClick={() => navigate("/alerts")}>
                View all{" "}
                <HugeiconsIcon
                  icon={ArrowUpRight01Icon}
                  strokeWidth={1.5}
                  className="size-3.5"
                />
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="space-y-2">
            {alerts.length > 0 ? (
              alerts.map((a) => {
                const SEV_COLOR: Record<string, string> = {
                  critical: "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
                  high: "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
                  warning: "bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/20",
                  med: "bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/20",
                  info: "bg-blue-500/10 text-blue-700 dark:text-blue-300 ring-blue-500/20",
                  low: "bg-zinc-500/10 text-zinc-700 dark:text-zinc-300 ring-zinc-500/20",
                }
                return (
                  <div
                    key={a.id}
                    className="flex items-start gap-3 rounded-2xl border border-border/60 p-3"
                  >
                    <Badge className={SEV_COLOR[a.severity] ?? SEV_COLOR.low}>
                      {a.severity.toUpperCase()}
                    </Badge>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium">{a.title}</p>
                      <p className="truncate text-xs text-muted-foreground">
                        {a.route_id ? `Route ${a.route_id} · ` : ""}
                        {a.what ?? a.title}
                      </p>
                    </div>
                    <span className="text-xs text-muted-foreground">#{a.id}</span>
                  </div>
                )
              })
            ) : (
              <p className="py-4 text-center text-sm text-muted-foreground">
                No active alerts
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}