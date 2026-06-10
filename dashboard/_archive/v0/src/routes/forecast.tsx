import { useQuery } from "@tanstack/react-query"
import { HugeiconsIcon } from "@hugeicons/react"
import { ChartIcon, RouteIcon, TargetIcon, TrendingUp } from "@hugeicons/core-free-icons"
import { useState } from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { ForecastChart } from "@/components/forecast-chart"
import { fetchForecastCompare, fetchPredictions, fetchRouteForecast, fetchRoutes } from "@/lib/api"
import { formatNumber } from "@/lib/utils"

export function ForecastPage() {
  const routes = useQuery({ queryKey: ["routes-forecast"], queryFn: fetchRoutes })
  const compare = useQuery({ queryKey: ["forecast-compare"], queryFn: () => fetchForecastCompare() })
  const predictions = useQuery({
    queryKey: ["predictions-forecast", 120],
    queryFn: () => fetchPredictions(120),
    refetchInterval: 30_000,
  })
  const [routeId, setRouteId] = useState<string | null>(null)
  const routeForecast = useQuery({
    queryKey: ["route-forecast", routeId],
    queryFn: () => fetchRouteForecast(routeId!),
    enabled: !!routeId,
  })

  const compareStations = compare.data?.stations ?? []
  const totals = compareStations.reduce((acc, s) => acc + (s.actual ?? 0), 0)
  const predictedTotal = compareStations.reduce((acc, s) => acc + s.predicted, 0)
  const overallAccuracy = totals > 0 ? Math.max(0, 1 - Math.abs(totals - predictedTotal) / Math.max(1, totals)) * 100 : null

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">Forecast</p>
        <h1 className="font-heading text-3xl font-medium tracking-tight">Predictive demand</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Model output for the next 2 hours, per-station error and route-level forecast drill-down.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-3">
        <ForecastChart
          title="2-hour forecast"
          data={(predictions.data?.predictions ?? []).slice(0, 24).map((p) => ({
            timestamp: p.timestamp,
            predicted: p.predicted,
            confidence: p.confidence,
          }))}
          loading={predictions.isLoading}
        />
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <span className="grid size-8 place-items-center rounded-2xl bg-muted text-foreground">
                <HugeiconsIcon icon={TargetIcon} strokeWidth={2} className="size-4" />
              </span>
              <div>
                <CardDescription>Aggregate accuracy</CardDescription>
                <CardTitle className="text-lg">Model performance</CardTitle>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="font-heading text-3xl font-medium tabular-nums">
              {overallAccuracy === null ? "—" : `${overallAccuracy.toFixed(1)}%`}
            </p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-2xl border border-border p-3">
                <p className="text-muted-foreground">Actual</p>
                <p className="font-heading text-base tabular-nums">{formatNumber(totals)}</p>
              </div>
              <div className="rounded-2xl border border-border p-3">
                <p className="text-muted-foreground">Predicted</p>
                <p className="font-heading text-base tabular-nums">{formatNumber(predictedTotal)}</p>
              </div>
            </div>
            <p className="rounded-2xl bg-muted p-3 text-xs text-muted-foreground">
              Compare {compareStations.length} stations · last refresh just now.
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <span className="grid size-8 place-items-center rounded-2xl bg-muted text-foreground">
                <HugeiconsIcon icon={TrendingUp} strokeWidth={2} className="size-4" />
              </span>
              <div>
                <CardDescription>Confidence</CardDescription>
                <CardTitle className="text-lg">Average band</CardTitle>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {predictions.isLoading ? (
              <Skeleton className="h-24 w-full rounded-2xl" />
            ) : (
              <ConfidenceBar
                values={(predictions.data?.predictions ?? []).map((p) => p.confidence).filter((v): v is number => v > 0)}
              />
            )}
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="stations">
        <TabsList>
          <TabsTrigger value="stations">Per-station</TabsTrigger>
          <TabsTrigger value="routes">By route</TabsTrigger>
        </TabsList>
        <TabsContent value="stations" className="space-y-3">
          {compare.isLoading ? (
            <Skeleton className="h-40 w-full rounded-3xl" />
          ) : (
            <Card>
              <CardContent className="p-0">
                <ul className="divide-y divide-border">
                  {compareStations.slice(0, 12).map((s) => {
                    const error = s.actual !== undefined ? Math.abs(s.actual - s.predicted) : null
                    const ratio = s.actual ? s.predicted / s.actual : null
                    return (
                      <li key={s.station_id} className="flex items-center gap-3 px-4 py-3 text-sm">
                        <span className="font-medium">{s.station_name}</span>
                        <span className="text-muted-foreground">pred {formatNumber(s.predicted)}</span>
                        <span className="text-muted-foreground">act {s.actual ?? "—"}</span>
                        <span className="ml-auto tabular-nums text-xs">
                          {error !== null ? `Δ ${formatNumber(error)}` : "—"}
                        </span>
                        <span
                          className={
                            ratio === null
                              ? "text-muted-foreground"
                              : Math.abs(1 - ratio) < 0.1
                                ? "text-chart-2"
                                : "text-chart-3"
                          }
                        >
                          {ratio === null ? "—" : `${(ratio * 100).toFixed(0)}%`}
                        </span>
                      </li>
                    )
                  })}
                </ul>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        <TabsContent value="routes" className="space-y-3">
          <Card>
            <CardHeader>
              <CardDescription>Routes</CardDescription>
              <CardTitle className="text-lg">Pick a route to inspect</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="flex flex-wrap gap-2">
                {(routes.data?.routes ?? []).map((r) => (
                  <li key={r.id}>
                    <button
                      type="button"
                      onClick={() => setRouteId(r.id)}
                      className={
                        "inline-flex items-center gap-2 rounded-2xl border px-3 py-1.5 text-sm transition-colors " +
                        (routeId === r.id
                          ? "border-foreground bg-foreground text-background"
                          : "border-border bg-card hover:bg-muted")
                      }
                    >
                      <HugeiconsIcon icon={RouteIcon} strokeWidth={2} className="size-3.5" />
                      {r.name}
                    </button>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
          {routeId && (
            <Card>
              <CardHeader>
                <CardDescription>Route forecast</CardDescription>
                <CardTitle className="text-lg">{routeId}</CardTitle>
              </CardHeader>
              <CardContent>
                {routeForecast.isLoading ? (
                  <Skeleton className="h-40 w-full rounded-2xl" />
                ) : (
                  <RouteForecastBars points={routeForecast.data?.forecast ?? []} />
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

function ConfidenceBar({ values }: { values: number[] }) {
  const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0
  return (
    <div className="space-y-2">
      <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-chart-2"
          style={{ width: `${Math.min(100, avg * 100)}%` }}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        {avg > 0 ? `${(avg * 100).toFixed(0)}% average confidence` : "No data"}
      </p>
    </div>
  )
}

function RouteForecastBars({
  points,
}: {
  points: { timestamp: string; predicted: number; confidence: number }[]
}) {
  if (points.length === 0) {
    return <p className="text-sm text-muted-foreground">No forecast available for this route.</p>
  }
  const max = Math.max(1, ...points.map((p) => p.predicted))
  return (
    <div className="space-y-2">
      <div className="flex h-40 items-end gap-1.5">
        {points.map((p, i) => {
          const h = (p.predicted / max) * 100
          return (
            <div
              key={`${p.timestamp}-${i}`}
              className="flex-1 rounded-t-xl bg-chart-1"
              style={{ height: `${h}%` }}
              title={`${p.timestamp} · ${p.predicted}`}
            />
          )
        })}
      </div>
      <p className="text-xs text-muted-foreground">
        <HugeiconsIcon icon={ChartIcon} strokeWidth={2} className="mr-1 inline-block size-3" />
        {points.length} steps · peak {Math.round(max)} passengers
      </p>
    </div>
  )
}

export default ForecastPage
