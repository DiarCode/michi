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
  Analytics01Icon,
  ArrowUpRight01Icon,
  ChartLineData01Icon,
  Download01Icon,
  Share01Icon,
} from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { KpiCard } from "@/components/kpi-card"

const KPIS = [
  { label: "Service hours · week", value: "12,408", delta: { value: "+2.4%", positive: true } },
  { label: "Riders · week", value: "312.4k", delta: { value: "+5.6%", positive: true } },
  { label: "Operating cost", value: "₸ 84.2M", delta: { value: "-1.1%", positive: true } },
  { label: "CO₂ avoided", value: "184 t", delta: { value: "+8.0%", positive: true } },
] as const

const ROUTES = [
  { id: "12", riders: "48.2k", ontime: "94%", cost: "₸ 6.4M", status: "Healthy" },
  { id: "22", riders: "32.1k", ontime: "88%", cost: "₸ 5.1M", status: "Watch" },
  { id: "47", riders: "27.0k", ontime: "82%", cost: "₸ 4.8M", status: "Watch" },
  { id: "08", riders: "21.7k", ontime: "96%", cost: "₸ 3.6M", status: "Healthy" },
  { id: "05", riders: "19.4k", ontime: "78%", cost: "₸ 3.9M", status: "At risk" },
] as const

const STATUS: Record<string, string> = {
  Healthy: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 ring-emerald-500/20",
  Watch: "bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/20",
  "At risk": "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
}

export function ExecutivePage() {
  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="Executive"
        title="Executive Overview"
        description="Weekly service health, financial snapshot, and the routes that need attention."
        actions={
          <>
            <Button variant="outline" size="sm">
              <HugeiconsIcon icon={Share01Icon} strokeWidth={1.5} className="size-3.5" />
              Share
            </Button>
            <Button size="sm">
              <HugeiconsIcon icon={Download01Icon} strokeWidth={1.5} className="size-3.5" />
              Export PDF
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {KPIS.map((k) => (
          <Card key={k.label} size="sm">
            <CardContent>
              <KpiCard {...k} icon={<HugeiconsIcon icon={Analytics01Icon} strokeWidth={1.5} className="size-3.5" />} />
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <Card>
          <CardHeader>
            <CardTitle>Route performance</CardTitle>
            <CardDescription>Top 5 routes · sorted by ridership</CardDescription>
            <CardAction>
              <Tabs defaultValue="week">
                <TabsList>
                  <TabsTrigger value="week">Week</TabsTrigger>
                  <TabsTrigger value="month">Month</TabsTrigger>
                  <TabsTrigger value="qtr">Quarter</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardAction>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Route</TableHead>
                  <TableHead>Riders</TableHead>
                  <TableHead>On-time</TableHead>
                  <TableHead>Cost</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ROUTES.map((r) => (
                  <TableRow key={r.id}>
                    <TableCell className="font-medium">Route {r.id}</TableCell>
                    <TableCell>{r.riders}</TableCell>
                    <TableCell>{r.ontime}</TableCell>
                    <TableCell>{r.cost}</TableCell>
                    <TableCell>
                      <Badge className={STATUS[r.status]}>{r.status}</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top insights</CardTitle>
            <CardDescription>Auto-generated from network signals</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {[
              {
                title: "Route 5 trending down",
                body: "On-time performance dropped 6pp week-over-week. Recommend a 4-week plan review.",
              },
              {
                title: "Bayterek corridor congestion",
                body: "Average dwell time up 28% during 17:00–18:30. Consider a turnback short-line.",
              },
              {
                title: "Energy efficiency up",
                body: "Regenerative braking usage improved 4.2% across the electric fleet.",
              },
            ].map((i) => (
              <div key={i.title} className="rounded-2xl border border-border/60 p-3">
                <p className="font-medium">{i.title}</p>
                <p className="text-sm text-muted-foreground">{i.body}</p>
                <Button variant="link" size="sm" className="px-0">
                  Read more <HugeiconsIcon icon={ArrowUpRight01Icon} strokeWidth={1.5} className="size-3.5" />
                </Button>
              </div>
            ))}
            <Tabs defaultValue="all">
              <TabsList>
                <TabsTrigger value="all">All</TabsTrigger>
                <TabsTrigger value="finance">Finance</TabsTrigger>
                <TabsTrigger value="ops">Ops</TabsTrigger>
              </TabsList>
              <TabsContent value="all" />
            </Tabs>
            <div className="flex items-center gap-2 rounded-2xl bg-muted/40 p-3 text-xs text-muted-foreground">
              <HugeiconsIcon icon={ChartLineData01Icon} strokeWidth={1.5} className="size-3.5" />
              Generated from last 7 days of synthetic data.
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
