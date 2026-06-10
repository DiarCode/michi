import { useQuery } from "@tanstack/react-query"
import { HugeiconsIcon } from "@hugeicons/react"
import { ActivityIcon, PauseIcon, PlayIcon, ShieldQuestionMarkIcon, TargetIcon, ZapIcon } from "@hugeicons/core-free-icons"
import { useEffect, useState } from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { KpiCard } from "@/components/kpi-card"
import { Skeleton } from "@/components/ui/skeleton"
import { fetchSimulationMetrics, fetchSimulationState, startSimulation, stopSimulation } from "@/lib/api"
import { toast } from "sonner"
import { useSimulationStore } from "@/stores/simulation-store"

export function SimulationPage() {
  const state = useQuery({
    queryKey: ["sim-state"],
    queryFn: fetchSimulationState,
    refetchInterval: 3_000,
  })
  const metrics = useQuery({
    queryKey: ["sim-metrics", 6],
    queryFn: () => fetchSimulationMetrics(6),
    refetchInterval: 5_000,
  })
  const setSim = useSimulationStore((s) => s.setState)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (state.data) {
      setSim({
        running: state.data.running,
        tick: state.data.tick,
        drift: (state.data.drift_status as "stable" | "drifting" | "critical") ?? "stable",
        lastUpdate: state.data.current_time ?? new Date().toISOString(),
      })
    }
  }, [state.data, setSim])

  const realtime = metrics.data?.realtime ?? []
  const db = metrics.data?.database ?? []
  const latest = realtime[realtime.length - 1]
  const accuracy = latest?.accuracy ?? null
  const mae = latest?.mae ?? null
  const mape = latest?.mape ?? null

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">Simulation</p>
          <h1 className="font-heading text-3xl font-medium tracking-tight">Digital twin of the network</h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Run synthetic traffic, evaluate model accuracy and watch for prediction drift in real time.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {state.data?.running ? (
            <Button
              variant="destructive"
              size="sm"
              disabled={busy}
              onClick={async () => {
                setBusy(true)
                try {
                  await stopSimulation()
                  toast.success("Simulation paused")
                } catch {
                  toast.error("Could not pause")
                } finally {
                  setBusy(false)
                }
              }}
            >
              <HugeiconsIcon icon={PauseIcon} strokeWidth={2} />
              <span>Pause</span>
            </Button>
          ) : (
            <Button
              size="sm"
              disabled={busy}
              onClick={async () => {
                setBusy(true)
                try {
                  await startSimulation()
                  toast.success("Simulation started")
                } catch {
                  toast.error("Could not start")
                } finally {
                  setBusy(false)
                }
              }}
            >
              <HugeiconsIcon icon={PlayIcon} strokeWidth={2} />
              <span>Start</span>
            </Button>
          )}
        </div>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="Tick"
          value={state.data?.tick ?? 0}
          icon={ActivityIcon}
          description="Simulation step"
        />
        <KpiCard
          title="Accuracy"
          value={accuracy}
          isPercent
          icon={TargetIcon}
          description="Last 5 minutes"
        />
        <KpiCard
          title="MAE"
          value={mae}
          icon={ZapIcon}
          description="Mean absolute error"
        />
        <KpiCard
          title="Drift"
          value={null}
          description={state.data?.drift_status ?? "—"}
          icon={ShieldQuestionMarkIcon}
        />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardDescription>Realtime</CardDescription>
                <CardTitle className="text-lg">Accuracy timeline</CardTitle>
              </div>
              <Badge variant={state.data?.running ? "default" : "secondary"} className="rounded-xl">
                {state.data?.running ? "Running" : "Idle"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            {realtime.length === 0 ? (
              <Skeleton className="h-40 w-full rounded-2xl" />
            ) : (
              <Sparkline
                values={realtime.map((r) => r.accuracy).filter((v): v is number => v !== null)}
                label="accuracy"
              />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardDescription>Drift</CardDescription>
            <CardTitle className="text-lg">Model status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span>Stability</span>
                <span className="tabular-nums">
                  {accuracy !== null ? Math.round(accuracy * 100) : "—"}%
                </span>
              </div>
              <Progress value={accuracy !== null ? accuracy * 100 : 0} />
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-2xl border border-border p-3">
                <p className="text-muted-foreground">MAE</p>
                <p className="font-heading text-base tabular-nums">{mae?.toFixed(2) ?? "—"}</p>
              </div>
              <div className="rounded-2xl border border-border p-3">
                <p className="text-muted-foreground">MAPE</p>
                <p className="font-heading text-base tabular-nums">
                  {mape !== null ? mape.toFixed(2) : "—"}
                </p>
              </div>
            </div>
            <p className="rounded-2xl bg-muted p-3 text-xs text-muted-foreground">
              {state.data?.drift_status === "critical"
                ? "Critical drift detected — model retraining recommended."
                : state.data?.drift_status === "drifting"
                  ? "Slight drift — monitor closely."
                  : "Model stable."}
            </p>
          </CardContent>
        </Card>
      </section>

      <Card>
        <CardHeader>
          <CardDescription>Database</CardDescription>
          <CardTitle className="text-lg">Recent metric snapshots</CardTitle>
        </CardHeader>
        <CardContent>
          {db.length === 0 ? (
            <Skeleton className="h-32 w-full rounded-2xl" />
          ) : (
            <ul className="space-y-2">
              {db.slice(0, 6).map((d, i) => (
                <li
                  key={`${d.timestamp}-${i}`}
                  className="flex items-center gap-3 rounded-2xl border border-border bg-card p-3 text-sm"
                >
                  <span className="font-medium tabular-nums">{new Date(d.timestamp).toLocaleTimeString()}</span>
                  <span className="text-muted-foreground">MAE {d.mae?.toFixed(2) ?? "—"}</span>
                  <span className="text-muted-foreground">MAPE {d.mape?.toFixed(2) ?? "—"}</span>
                  <span className="ml-auto text-muted-foreground">{d.count} samples</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function Sparkline({ values, label }: { values: number[]; label: string }) {
  if (values.length === 0) {
    return <p className="text-sm text-muted-foreground">No data yet.</p>
  }
  const max = Math.max(1, ...values)
  const min = Math.min(0, ...values)
  const range = max - min || 1
  const width = 480
  const height = 120
  const stepX = values.length > 1 ? width / (values.length - 1) : width
  const points = values.map((v, i) => {
    const x = i * stepX
    const y = height - ((v - min) / range) * (height - 8) - 4
    return `${x},${y}`
  })
  return (
    <div className="space-y-2">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-32 w-full" preserveAspectRatio="none">
        <polyline
          points={points.join(" ")}
          fill="none"
          stroke="var(--chart-1)"
          strokeWidth="2"
          strokeLinejoin="round"
        />
      </svg>
      <p className="text-xs text-muted-foreground">{label} · {values.length} samples</p>
    </div>
  )
}

export default SimulationPage
