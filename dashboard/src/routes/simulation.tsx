import { useState } from "react"
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
import { ArrowReloadHorizontalIcon, PlayIcon, StopIcon } from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { useSimulationStore } from "@/stores/simulation-store"
import { KpiCard } from "@/components/kpi-card"
import { Progress } from "@/components/ui/progress"

export function SimulationPage() {
  const { running, startSimulation, stopSimulation } = useSimulationStore()
  const setRunning = (v: boolean) => (v ? startSimulation() : stopSimulation())
  const speed = 1
  const setSpeed = (_: number) => {}
  const reset = () => {}
  const [horizon, setHorizon] = useState(60)

  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="What-if"
        title="Simulation"
        description="Run counterfactual scenarios against the DTS-GSSF model. Compare interventions before deployment."
        actions={
          <>
            {running ? (
              <Button variant="outline" size="sm" onClick={() => setRunning(false)}>
                <HugeiconsIcon icon={StopIcon} strokeWidth={1.5} className="size-3.5" />
                Pause
              </Button>
            ) : (
              <Button size="sm" onClick={() => setRunning(true)}>
                <HugeiconsIcon icon={PlayIcon} strokeWidth={1.5} className="size-3.5" />
                Start
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={reset}>
              <HugeiconsIcon icon={ArrowReloadHorizontalIcon} strokeWidth={1.5} className="size-3.5" />
              Reset
            </Button>
          </>
        }
      />

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent>
            <KpiCard label="Sim clock" value="00:00:00" hint="Synthetic time" />
            <Progress value={running ? 40 : 0} className="mt-3 h-1" />
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <KpiCard label="Speed" value={`${speed.toFixed(1)}x`} hint="Wall-clock multiplier" />
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <KpiCard label="Horizon" value={`${horizon}m`} hint="Forecast window" />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <Card>
          <CardHeader>
            <CardTitle>Scenario</CardTitle>
            <CardDescription>Tweak simulation parameters and rerun the model.</CardDescription>
            <CardAction>
              <Badge variant="secondary">DTS-GSSF · horizon 4</Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Sim speed</Label>
                <span className="text-xs text-muted-foreground">{speed.toFixed(1)}x</span>
              </div>
              <Slider min={0.5} max={4} step={0.5} value={[speed]} onValueChange={(v) => setSpeed(v[0])} />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Horizon (minutes)</Label>
                <span className="text-xs text-muted-foreground">{horizon} min</span>
              </div>
              <Slider min={15} max={180} step={15} value={[horizon]} onValueChange={(v) => setHorizon(v[0])} />
            </div>

            <div className="space-y-3 rounded-2xl bg-muted/40 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="reroute">Auto-reroute on alert</Label>
                  <p className="text-xs text-muted-foreground">Trigger reroute playbooks automatically.</p>
                </div>
                <Switch id="reroute" defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="emit">Emit metrics to bus</Label>
                  <p className="text-xs text-muted-foreground">Stream synthetic metrics over WebSocket.</p>
                </div>
                <Switch id="emit" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Run log</CardTitle>
            <CardDescription>Last 5 events</CardDescription>
          </CardHeader>
          <CardContent>
            <ol className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-emerald-500" />
                Loaded bundle · 184 vehicles
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-emerald-500" />
                Z-score normalization applied
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-amber-500" />
                Mock backend unavailable — using synthetic stream
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-zinc-400" />
                Waiting for first tick…
              </li>
            </ol>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
