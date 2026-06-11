import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { SectionHeader } from "@/components/section-header"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  ArrowReloadHorizontalIcon,
} from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Field,
  FieldGroup,
  FieldLabel,
  FieldDescription,
} from "@/components/ui/field"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { fetchPredictions, fetchStations, fetchForecastCompare } from "@/lib/api"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts"

const HORIZON_MAP: Record<string, number> = {
  "15m": 15,
  "60m": 60,
  "4h": 240,
  "24h": 1440,
}

export function ForecastPage() {
  const [horizon, setHorizon] = useState("60m")
  const [model, setModel] = useState("dts-gssf")
  const [stationId, setStationId] = useState("__all__")

  const horizonMin = HORIZON_MAP[horizon] ?? 60

  const { data: stationsData } = useQuery({
    queryKey: ["stations-forecast"],
    queryFn: () => fetchStations(),
    staleTime: 5 * 60 * 1000,
  })

  const { data: predictions, isLoading: predLoading, isError: predError, refetch } = useQuery({
    queryKey: ["predictions", horizonMin],
    queryFn: () => fetchPredictions(horizonMin),
    refetchInterval: 300_000,
  })

  const { data: compareData, isLoading: compareLoading } = useQuery({
    queryKey: ["forecast-compare", stationId === "__all__" ? undefined : stationId],
    queryFn: () => fetchForecastCompare(stationId === "__all__" ? undefined : stationId),
    enabled: model === "dts-gssf",
  })

  const stations = stationsData?.stations ?? []

  // Group predictions by station for charting
  const chartData = useMemo(() => {
    if (!predictions?.predictions) return []
    const grouped: Record<string, Record<string, { predicted: number; confidence: number }>> = {}
    for (const p of predictions.predictions) {
      const key = p.station_id
      if (!grouped[key]) grouped[key] = {}
      grouped[key][p.horizon_minutes] = { predicted: p.predicted, confidence: p.confidence }
    }
    // Build chart-friendly array from the first station or selected station
    const targetId = stationId === "__all__"
      ? Object.keys(grouped)[0]
      : stationId
    const stationPreds = grouped[targetId ?? ""]
    if (!stationPreds) return []
    return Object.entries(stationPreds)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([min, val]) => ({
        horizon: `${min}m`,
        predicted: Math.round(val.predicted),
        confidence: Math.round(val.confidence * 100),
      }))
  }, [predictions, stationId])

  // Summary stats
  const summary = useMemo(() => {
    if (!predictions?.predictions || predictions.predictions.length === 0) {
      return { totalRiders: 0, peakLoad: 0, avgConfidence: 0 }
    }
    const preds = predictions.predictions
    const totalRiders = preds.reduce((s, p) => s + p.predicted, 0)
    const peakLoad = Math.max(...preds.map((p) => p.predicted))
    const avgConfidence = preds.reduce((s, p) => s + p.confidence, 0) / preds.length
    return {
      totalRiders: Math.round(totalRiders),
      peakLoad,
      avgConfidence: Math.round(avgConfidence * 100),
    }
  }, [predictions])

  // Compare data for baseline tab
  const compareChartData = useMemo(() => {
    if (!compareData?.models) return []
    // Merge all model forecasts into one chart
    const allHorizons = new Set<number>()
    for (const m of compareData.models) {
      for (const f of m.forecast) allHorizons.add(f.hour)
    }
    return Array.from(allHorizons).sort((a, b) => a - b).map((h) => {
      const entry: Record<string, number> = { hour: h }
      for (const m of compareData!.models) {
        const point = m.forecast.find((f) => f.hour === h)
        entry[m.name] = point?.predicted ?? 0
      }
      return entry
    })
  }, [compareData])

  const modelColors = ["#3b82f6", "#f59e0b", "#10b981"]

  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="Predict"
        title="Forecast"
        description="Generate a passenger-flow forecast from the DTS-GSSF model. Compare against baselines."
        actions={
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetch()}
              disabled={predLoading}
            >
              <HugeiconsIcon
                icon={ArrowReloadHorizontalIcon}
                strokeWidth={1.5}
                className="size-3.5"
              />
              Refresh
            </Button>
          </>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[22rem_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>
              Choose inputs and the model to run.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FieldGroup>
              <Field>
                <FieldLabel>Forecast horizon</FieldLabel>
                <ToggleGroup
                  type="single"
                  value={horizon}
                  onValueChange={(v) => v && setHorizon(v)}
                  variant="outline"
                  size="sm"
                  className="w-full"
                >
                  <ToggleGroupItem value="15m" className="flex-1">
                    15m
                  </ToggleGroupItem>
                  <ToggleGroupItem value="60m" className="flex-1">
                    60m
                  </ToggleGroupItem>
                  <ToggleGroupItem value="4h" className="flex-1">
                    4h
                  </ToggleGroupItem>
                  <ToggleGroupItem value="24h" className="flex-1">
                    24h
                  </ToggleGroupItem>
                </ToggleGroup>
              </Field>

              <Field>
                <FieldLabel>Model</FieldLabel>
                <ToggleGroup
                  type="single"
                  value={model}
                  onValueChange={(v) => v && setModel(v)}
                  variant="outline"
                  size="sm"
                  className="w-full"
                >
                  <ToggleGroupItem value="dts-gssf" className="flex-1">
                    DTS-GSSF
                  </ToggleGroupItem>
                  <ToggleGroupItem value="stgcn" className="flex-1">
                    STGCN
                  </ToggleGroupItem>
                  <ToggleGroupItem value="dcrnn" className="flex-1">
                    DCRNN
                  </ToggleGroupItem>
                </ToggleGroup>
                <FieldDescription>
                  Default is the paper's primary model.
                </FieldDescription>
              </Field>

              <Field>
                <Label htmlFor="station-select">Station</Label>
                <Select value={stationId} onValueChange={setStationId}>
                  <SelectTrigger id="station-select">
                    <SelectValue placeholder="All stations" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__all__">All stations</SelectItem>
                    {stations.map((s) => (
                      <SelectItem key={s.id} value={s.id}>
                        {s.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            </FieldGroup>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <Tabs defaultValue="ridership">
                <TabsList>
                  <TabsTrigger value="ridership">Ridership</TabsTrigger>
                  <TabsTrigger value="headway">Headway</TabsTrigger>
                  <TabsTrigger value="anomaly">Anomaly</TabsTrigger>
                </TabsList>
              </Tabs>
              <CardAction>
                {predictions?.predictions?.[0] && (
                  <Badge>
                    {predictions.predictions[0].model_version} ·{" "}
                    {predictions.predictions.length} predictions
                  </Badge>
                )}
              </CardAction>
            </CardHeader>
            <CardContent>
              {predLoading ? (
                <Skeleton className="aspect-[16/6] w-full rounded-2xl" />
              ) : predError ? (
                <div className="grid aspect-[16/6] place-items-center rounded-2xl border border-dashed border-border/60 bg-muted/30 text-sm text-muted-foreground">
                  Failed to load forecast data. Try refreshing.
                </div>
              ) : chartData.length > 0 ? (
                <div className="aspect-[16/6]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" />
                      <XAxis dataKey="horizon" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{
                          borderRadius: 12,
                          border: "1px solid var(--border)",
                          background: "var(--popover)",
                          color: "var(--popover-foreground)",
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        name="Predicted ridership"
                      />
                      <Line
                        type="monotone"
                        dataKey="confidence"
                        stroke="#10b981"
                        strokeWidth={1.5}
                        strokeDasharray="5 5"
                        dot={false}
                        name="Confidence %"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="grid aspect-[16/6] place-items-center rounded-2xl border border-dashed border-border/60 bg-muted/30 text-sm text-muted-foreground">
                  No prediction data available for this horizon
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Summary</CardTitle>
              <CardDescription>
                Across all selected routes for the next{" "}
                {horizon}.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="ridership">
                <TabsList>
                  <TabsTrigger value="ridership">Ridership</TabsTrigger>
                  <TabsTrigger value="baseline">vs Baseline</TabsTrigger>
                </TabsList>
                <TabsContent value="ridership" className="mt-3">
                  <ul className="grid gap-2 text-sm sm:grid-cols-3">
                    <li className="rounded-2xl bg-muted/40 p-3">
                      <p className="text-xs text-muted-foreground">
                        Expected riders
                      </p>
                      <p className="mt-1 font-heading text-xl font-medium">
                        {summary.totalRiders.toLocaleString()}
                      </p>
                    </li>
                    <li className="rounded-2xl bg-muted/40 p-3">
                      <p className="text-xs text-muted-foreground">Peak load</p>
                      <p className="mt-1 font-heading text-xl font-medium">
                        {summary.peakLoad.toLocaleString()}
                      </p>
                    </li>
                    <li className="rounded-2xl bg-muted/40 p-3">
                      <p className="text-xs text-muted-foreground">
                        Avg confidence
                      </p>
                      <p className="mt-1 font-heading text-xl font-medium">
                        {summary.avgConfidence}%
                      </p>
                    </li>
                  </ul>
                </TabsContent>
                <TabsContent value="baseline" className="mt-3">
                  {compareLoading ? (
                    <Skeleton className="h-48 w-full rounded-2xl" />
                  ) : compareChartData.length > 0 ? (
                    <div className="aspect-[16/8]">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={compareChartData}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" />
                          <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                          <YAxis tick={{ fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{
                              borderRadius: 12,
                              border: "1px solid var(--border)",
                              background: "var(--popover)",
                              color: "var(--popover-foreground)",
                            }}
                          />
                          <Legend />
                          {compareData?.models.map((m, i) => (
                            <Line
                              key={m.name}
                              type="monotone"
                              dataKey={m.name}
                              stroke={modelColors[i % modelColors.length]}
                              strokeWidth={2}
                              dot={{ r: 2 }}
                              name={m.name.toUpperCase()}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No baseline comparison data available yet. Run the DTS-GSSF model to generate comparisons.
                    </p>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}