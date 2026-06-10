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
import { Calendar01Icon, ChartLineData01Icon, Download01Icon } from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Field, FieldGroup, FieldLabel, FieldDescription } from "@/components/ui/field"

export function ForecastPage() {
  const [horizon, setHorizon] = useState("60m")
  const [model, setModel] = useState("dts-gssf")

  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="Predict"
        title="Forecast"
        description="Generate a 60-minute passenger-flow forecast from the DTS-GSSF model. Compare against baselines."
        actions={
          <>
            <Button variant="outline" size="sm">
              <HugeiconsIcon icon={Download01Icon} strokeWidth={1.5} className="size-3.5" />
              Export
            </Button>
            <Button size="sm">
              <HugeiconsIcon icon={ChartLineData01Icon} strokeWidth={1.5} className="size-3.5" />
              Run forecast
            </Button>
          </>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[22rem_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>Choose inputs and the model to run.</CardDescription>
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
                  <ToggleGroupItem value="15m" className="flex-1">15m</ToggleGroupItem>
                  <ToggleGroupItem value="60m" className="flex-1">60m</ToggleGroupItem>
                  <ToggleGroupItem value="4h" className="flex-1">4h</ToggleGroupItem>
                  <ToggleGroupItem value="24h" className="flex-1">24h</ToggleGroupItem>
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
                  <ToggleGroupItem value="dts-gssf" className="flex-1">DTS-GSSF</ToggleGroupItem>
                  <ToggleGroupItem value="stgcn" className="flex-1">STGCN</ToggleGroupItem>
                  <ToggleGroupItem value="dcrnn" className="flex-1">DCRNN</ToggleGroupItem>
                </ToggleGroup>
                <FieldDescription>Default is the paper's primary model.</FieldDescription>
              </Field>

              <Field>
                <Label htmlFor="from">From</Label>
                <div className="relative">
                  <HugeiconsIcon
                    icon={Calendar01Icon}
                    strokeWidth={1.5}
                    className="absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
                  />
                  <Input id="from" type="datetime-local" className="pl-8" />
                </div>
              </Field>

              <Field>
                <Label htmlFor="routes">Routes</Label>
                <Input id="routes" placeholder="e.g. 12, 47, 22" />
                <FieldDescription>Comma-separated route ids.</FieldDescription>
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
                <Badge>R² · 0.885</Badge>
              </CardAction>
            </CardHeader>
            <CardContent>
              <div className="grid aspect-[16/6] place-items-center rounded-2xl border border-dashed border-border/60 bg-muted/30 text-sm text-muted-foreground">
                Forecast chart placeholder · horizon {horizon} · {model}
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="ridership">
            <TabsContent value="ridership" />
          </Tabs>

          <Card>
            <CardHeader>
              <CardTitle>Summary</CardTitle>
              <CardDescription>Across all selected routes for the next hour.</CardDescription>
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
                      <p className="text-xs text-muted-foreground">Expected riders</p>
                      <p className="mt-1 font-heading text-xl font-medium">12,840</p>
                    </li>
                    <li className="rounded-2xl bg-muted/40 p-3">
                      <p className="text-xs text-muted-foreground">Peak load</p>
                      <p className="mt-1 font-heading text-xl font-medium">0.78</p>
                    </li>
                    <li className="rounded-2xl bg-muted/40 p-3">
                      <p className="text-xs text-muted-foreground">MAE</p>
                      <p className="mt-1 font-heading text-xl font-medium">2.20</p>
                    </li>
                  </ul>
                </TabsContent>
                <TabsContent value="baseline" className="mt-3">
                  <p className="text-sm text-muted-foreground">Baseline comparison data will render here.</p>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
