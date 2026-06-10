import { useQuery } from "@tanstack/react-query"
import { HugeiconsIcon } from "@hugeicons/react"
import { ActivityIcon, MapPinIcon, Shield01Icon, TargetIcon, TrendingUp, ZapIcon } from "@hugeicons/core-free-icons"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { KpiCard } from "@/components/kpi-card"
import { ForecastChart } from "@/components/forecast-chart"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { fetchKPIs, fetchRichAlerts, fetchRoutes, fetchStations, fetchPredictions } from "@/lib/api"
import { formatRelativeTime } from "@/lib/utils"

export function CommandCenterPage() {
  const kpis = useQuery({ queryKey: ["kpis"], queryFn: fetchKPIs })
  const alerts = useQuery({ queryKey: ["rich-alerts"], queryFn: fetchRichAlerts, refetchInterval: 15_000 })
  const routes = useQuery({ queryKey: ["routes"], queryFn: fetchRoutes })
  const stations = useQuery({ queryKey: ["stations"], queryFn: () => fetchStations() })
  const predictions = useQuery({
    queryKey: ["predictions", 60],
    queryFn: () => fetchPredictions(60),
    refetchInterval: 30_000,
  })

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">Command Center</p>
        <h1 className="font-heading text-3xl font-medium tracking-tight">Astana at a glance</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Real-time operational health, predictive signals and intervention readiness for the Astana bus network.
        </p>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="Active Stations"
          value={kpis.data?.total_stations ?? null}
          delta={0.8}
          icon={MapPinIcon}
          description="Network coverage"
          loading={kpis.isLoading}
        />
        <KpiCard
          title="Active Routes"
          value={kpis.data?.active_routes ?? null}
          delta={-0.3}
          icon={TargetIcon}
          description="In service right now"
          loading={kpis.isLoading}
        />
        <KpiCard
          title="Avg Ridership"
          value={kpis.data?.avg_ridership ?? null}
          delta={4.2}
          icon={TrendingUp}
          description="Passengers per hour"
          loading={kpis.isLoading}
        />
        <KpiCard
          title="On-time Performance"
          value={kpis.data?.on_time_performance ?? null}
          isPercent
          delta={1.1}
          icon={ActivityIcon}
          description="Past 24 hours"
          loading={kpis.isLoading}
        />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <ForecastChart
          title="60-minute Prediction"
          data={(predictions.data?.predictions ?? []).slice(0, 12).map((p) => ({
            timestamp: p.timestamp,
            predicted: p.predicted,
            confidence: p.confidence,
          }))}
          loading={predictions.isLoading}
        />
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardDescription>Active Alerts</CardDescription>
                <CardTitle className="text-lg">What needs attention</CardTitle>
              </div>
              <Button size="sm" variant="outline" className="rounded-2xl">
                <HugeiconsIcon icon={ZapIcon} strokeWidth={2} />
                <span>Acknowledge all</span>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {alerts.isLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-16 w-full rounded-2xl" />
                <Skeleton className="h-16 w-full rounded-2xl" />
                <Skeleton className="h-16 w-full rounded-2xl" />
              </div>
            ) : (alerts.data?.alerts ?? []).length === 0 ? (
              <div className="grid place-items-center rounded-2xl border border-dashed border-border py-10 text-center">
                <span className="grid size-10 place-items-center rounded-full bg-muted">
                  <HugeiconsIcon icon={Shield01Icon} strokeWidth={2} className="size-5 text-chart-2" />
                </span>
                <p className="mt-3 font-medium">All clear</p>
                <p className="text-xs text-muted-foreground">No active alerts on the network.</p>
              </div>
            ) : (
              <ul className="space-y-2">
                {(alerts.data?.alerts ?? []).slice(0, 5).map((a) => (
                  <li
                    key={a.id}
                    className="flex items-start gap-3 rounded-2xl border border-border bg-card p-3"
                  >
                    <span
                      className={
                        a.severity === "critical"
                          ? "mt-1 size-2.5 rounded-full bg-destructive"
                          : a.severity === "warning"
                            ? "mt-1 size-2.5 rounded-full bg-chart-3"
                            : "mt-1 size-2.5 rounded-full bg-muted-foreground"
                      }
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <p className="truncate font-medium">{a.title}</p>
                        <span className="ml-auto text-xs text-muted-foreground">
                          {formatRelativeTime(a.created_at)}
                        </span>
                      </div>
                      <p className="truncate text-xs text-muted-foreground">{a.message}</p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardDescription>Stations</CardDescription>
            <CardTitle className="text-lg">Top of the network</CardTitle>
          </CardHeader>
          <CardContent>
            {stations.isLoading ? (
              <Skeleton className="h-40 w-full rounded-2xl" />
            ) : (
              <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {(stations.data?.stations ?? []).slice(0, 9).map((s) => (
                  <li
                    key={s.id}
                    className="flex items-center gap-2 rounded-2xl border border-border bg-card p-3"
                  >
                    <span className="grid size-8 place-items-center rounded-xl bg-muted text-foreground">
                      <HugeiconsIcon icon={MapPinIcon} strokeWidth={2} className="size-4" />
                    </span>
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{s.name}</p>
                      <p className="truncate text-xs text-muted-foreground">{s.district ?? s.id}</p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardDescription>Routes</CardDescription>
            <CardTitle className="text-lg">Service health</CardTitle>
          </CardHeader>
          <CardContent>
            {routes.isLoading ? (
              <Skeleton className="h-40 w-full rounded-2xl" />
            ) : (
              <ul className="space-y-2">
                {(routes.data?.routes ?? []).slice(0, 8).map((r) => (
                  <li
                    key={r.id}
                    className="flex items-center gap-3 rounded-2xl border border-border bg-card p-3"
                  >
                    <span
                      className="size-3 shrink-0 rounded-full"
                      style={{ background: r.color ?? "var(--chart-1)" }}
                    />
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium">{r.name}</p>
                      <p className="truncate text-xs text-muted-foreground">{r.short_name ?? r.id}</p>
                    </div>
                    <span className="text-xs text-chart-2">healthy</span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

export default CommandCenterPage
