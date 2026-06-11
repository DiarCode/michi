import { useState, useEffect, useRef } from "react"
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
  PlayIcon,
  StopIcon,
} from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { useSimulationStore } from "@/stores/simulation-store"
import { useQuery, useMutation } from "@tanstack/react-query"
import {
  startSimulation as apiStartSimulation,
  stopSimulation as apiStopSimulation,
  fetchSimulationState,
  fetchSimulationMetrics,
} from "@/lib/api"
import { showToast } from "@/lib/toast"
import { KpiCard } from "@/components/kpi-card"
import { Progress } from "@/components/ui/progress"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts"

export function SimulationPage() {
  const {
    running,
    tick,
    metricsHistory,
    driftAlerts,
    startSimulation,
    stopSimulation,
    subscribe,
  } = useSimulationStore()

  const [speed, setSpeed] = useState(1)
  const [horizon, setHorizon] = useState(60)
  const [autoReroute, setAutoReroute] = useState(true)
  const [emitMetrics, setEmitMetrics] = useState(false)
  const unsubRef = useRef<(() => void) | null>(null)

  // Subscribe to WS events on mount
  useEffect(() => {
    const unsub = subscribe()
    unsubRef.current = unsub
    return () => {
      unsubRef.current?.()
      unsubRef.current = null
    }
  }, [subscribe])

  // Fetch current simulation state on mount
  const { data: simState } = useQuery({
    queryKey: ["simulation-state"],
    queryFn: fetchSimulationState,
    refetchInterval: 10_000,
  })

  // Fetch historical metrics
  useQuery({
    queryKey: ["simulation-metrics"],
    queryFn: () => fetchSimulationMetrics(1),
    refetchInterval: 30_000,
  })

  const startMut = useMutation({
    mutationFn: apiStartSimulation,
    onSuccess: () => {
      startSimulation()
      showToast.success("Simulation started")
    },
    onError: () => showToast.error("Failed to start simulation"),
  })

  const stopMut = useMutation({
    mutationFn: apiStopSimulation,
    onSuccess: () => {
      stopSimulation()
      showToast.success("Simulation stopped")
    },
    onError: () => {
      stopSimulation()
      showToast.error("Simulation stopped (backend unreachable)")
    },
  })

  const handleStartStop = () => {
    if (running) {
      stopMut.mutate()
    } else {
      startMut.mutate()
    }
  }

  const handleReset = () => {
    if (running) {
      stopMut.mutate()
    }
    useSimulationStore.setState({
      running: false,
      tick: 0,
      startTime: null,
      metricsHistory: [],
      driftAlerts: [],
      isStale: false,
      lastTickAt: null,
    })
    showToast.success("Simulation reset")
  }

  // Latest metric
  const latestMetric = metricsHistory.length > 0
    ? metricsHistory[metricsHistory.length - 1]
    : null

  const currentAccuracy = latestMetric?.accuracy ?? simState?.metrics?.accuracy ?? null
  const currentMape = latestMetric?.mape ?? simState?.metrics?.mape ?? null
  const driftStatus = useSimulationStore((s) => s.driftAlerts.length > 0
    ? s.driftAlerts[s.driftAlerts.length - 1].severity
    : (simState?.drift_status ?? "normal"))

  // Chart data
  const chartData = metricsHistory.slice(-60).map((m, i) => ({
    tick: m.tick ?? i,
    mae: m.mae,
    mape: Math.round(m.mape * 10) / 10,
    accuracy: m.accuracy ?? Math.max(0, 100 - m.mape),
  }))

  // Run log entries
  const runLog = [
    ...(running ? [{ text: "Simulation running", status: "emerald" as const }] : []),
    ...(latestMetric ? [{ text: `Tick ${latestMetric.tick ?? tick}: MAE ${latestMetric.mae?.toFixed(1)}, Accuracy ${currentAccuracy?.toFixed(1)}%`, status: "emerald" as const }] : []),
    ...(driftAlerts.length > 0 ? [{ text: `Drift alert: ${driftAlerts[driftAlerts.length - 1].metric} — ${driftAlerts[driftAlerts.length - 1].severity}`, status: "amber" as const }] : []),
    ...(simState ? [{ text: `Backend: ${simState.station_count ?? 0} stations, ${simState.running ? "active" : "idle"}`, status: "emerald" as const }] : [{ text: "Connecting to simulation backend…", status: "zinc" as const }]),
  ]

  // Sim clock display
  const simClock = simState?.current_time
    ? new Date(simState.current_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : tick > 0
      ? `Tick ${tick}`
      : "00:00:00"

  const progressValue = running ? Math.min(100, tick * 2) : 0

  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="What-if"
        title="Simulation"
        description="Run counterfactual scenarios against the DTS-GSSF model. Compare interventions before deployment."
        actions={
          <>
            {running ? (
              <Button
                variant="outline"
                size="sm"
                onClick={handleStartStop}
                disabled={stopMut.isPending}
              >
                <HugeiconsIcon
                  icon={StopIcon}
                  strokeWidth={1.5}
                  className="size-3.5"
                />
                Pause
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={handleStartStop}
                disabled={startMut.isPending}
              >
                <HugeiconsIcon
                  icon={PlayIcon}
                  strokeWidth={1.5}
                  className="size-3.5"
                />
                Start
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={handleReset}>
              <HugeiconsIcon
                icon={ArrowReloadHorizontalIcon}
                strokeWidth={1.5}
                className="size-3.5"
              />
              Reset
            </Button>
          </>
        }
      />

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent>
            <KpiCard label="Sim clock" value={simClock} hint={running ? "Live" : "Paused"} />
            <Progress value={progressValue} className="mt-3 h-1" />
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <KpiCard
              label="Accuracy"
              value={currentAccuracy != null ? `${currentAccuracy.toFixed(1)}%` : "—"}
              hint={driftStatus === "normal" ? "Normal" : driftStatus === "warning" ? "Warning" : "Critical"}
            />
            {currentMape != null && (
              <p className="mt-1 text-xs text-muted-foreground">
                MAPE: {currentMape.toFixed(1)}%
              </p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <KpiCard
              label="Speed"
              value={`${speed.toFixed(1)}x`}
              hint="Wall-clock multiplier"
            />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <Card>
          <CardHeader>
            <CardTitle>Validation Metrics</CardTitle>
            <CardDescription>
              Real-time model performance during simulation
            </CardDescription>
            <CardAction>
              <Badge variant="secondary">
                DTS-GSSF · {metricsHistory.length} ticks
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            {chartData.length > 1 ? (
              <div className="aspect-[16/6]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" />
                    <XAxis dataKey="tick" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        borderRadius: 12,
                        border: "1px solid var(--border)",
                        background: "var(--popover)",
                        color: "var(--popover-foreground)",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      name="Accuracy %"
                    />
                    <Line
                      type="monotone"
                      dataKey="mae"
                      stroke="#f59e0b"
                      strokeWidth={1.5}
                      dot={false}
                      name="MAE"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="grid aspect-[16/6] place-items-center rounded-2xl border border-dashed border-border/60 bg-muted/30 text-sm text-muted-foreground">
                {running
                  ? "Waiting for first validation metric…"
                  : "Start the simulation to see validation metrics"}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
              <CardDescription>Tweak simulation parameters.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Sim speed</Label>
                  <span className="text-xs text-muted-foreground">
                    {speed.toFixed(1)}x
                  </span>
                </div>
                <Slider
                  min={0.5}
                  max={4}
                  step={0.5}
                  value={[speed]}
                  onValueChange={(v) => setSpeed(v[0])}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Horizon (minutes)</Label>
                  <span className="text-xs text-muted-foreground">
                    {horizon} min
                  </span>
                </div>
                <Slider
                  min={15}
                  max={180}
                  step={15}
                  value={[horizon]}
                  onValueChange={(v) => setHorizon(v[0])}
                />
              </div>

              <div className="space-y-3 rounded-2xl bg-muted/40 p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="reroute">Auto-reroute on alert</Label>
                    <p className="text-xs text-muted-foreground">
                      Trigger reroute playbooks automatically.
                    </p>
                  </div>
                  <Switch
                    id="reroute"
                    checked={autoReroute}
                    onCheckedChange={setAutoReroute}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="emit">Emit metrics to bus</Label>
                    <p className="text-xs text-muted-foreground">
                      Stream synthetic metrics over WebSocket.
                    </p>
                  </div>
                  <Switch
                    id="emit"
                    checked={emitMetrics}
                    onCheckedChange={setEmitMetrics}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Run log</CardTitle>
              <CardDescription>Last {runLog.length} events</CardDescription>
            </CardHeader>
            <CardContent>
              {runLog.length > 0 ? (
                <ol className="space-y-2 text-sm">
                  {runLog.map((entry, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <span
                        className={`size-1.5 rounded-full ${
                          entry.status === "emerald"
                            ? "bg-emerald-500"
                            : entry.status === "amber"
                              ? "bg-amber-500"
                              : "bg-zinc-400"
                        }`}
                      />
                      {entry.text}
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Start the simulation to see events.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}