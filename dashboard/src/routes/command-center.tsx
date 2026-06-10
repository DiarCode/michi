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
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

const KPIS = [
  { label: "Active buses", value: "184", delta: { value: "+3.2%", positive: true }, icon: <HugeiconsIcon icon={Bus01Icon} strokeWidth={1.5} className="size-3.5" /> },
  { label: "On-time", value: "92.4%", delta: { value: "+0.6%", positive: true }, icon: <HugeiconsIcon icon={Time01Icon} strokeWidth={1.5} className="size-3.5" /> },
  { label: "Open alerts", value: "07", delta: { value: "-12%", positive: true }, icon: <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.5} className="size-3.5" /> },
  { label: "Riders · today", value: "48.2k", delta: { value: "+5.1%", positive: true }, icon: <HugeiconsIcon icon={UserGroupIcon} strokeWidth={1.5} className="size-3.5" /> },
]

const ALERTS = [
  { id: "A-201", severity: "high", route: "Route 12", text: "Heavy congestion near Bayterek — delay +6 min" },
  { id: "A-202", severity: "med", route: "Route 47", text: "Vehicle substitution required at Stop 3" },
  { id: "A-203", severity: "low", route: "Route 08", text: "Stop request outside schedule window" },
  { id: "A-204", severity: "med", route: "Route 22", text: "Door sensor intermittently offline" },
]

const SEV_COLOR: Record<string, string> = {
  high: "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
  med: "bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/20",
  low: "bg-zinc-500/10 text-zinc-700 dark:text-zinc-300 ring-zinc-500/20",
}

const CREW = [
  { initials: "AK", name: "Aigerim K.", role: "Operator · L1" },
  { initials: "DM", name: "Daniyar M.", role: "Operator · L2" },
  { initials: "SB", name: "Saltanat B.", role: "Dispatcher" },
  { initials: "YK", name: "Yerlan K.", role: "Maintenance" },
]

export function CommandCenterPage() {
  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Operations"
        title="Command Center"
        description="Live view of the Astana bus network. Track KPIs, respond to alerts, and review recent operator activity."
        actions={
          <>
            <Button variant="outline" size="sm">
              <HugeiconsIcon icon={FlashIcon} strokeWidth={1.5} className="size-3.5" />
              Run playbook
            </Button>
            <Button size="sm">
              <HugeiconsIcon icon={ChartLineData01Icon} strokeWidth={1.5} className="size-3.5" />
              Forecast
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {KPIS.map((k) => (
          <Card key={k.label} size="sm">
            <CardContent>
              <KpiCard {...k} />
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Network health</CardTitle>
            <CardDescription>Last 24 hours · synthetic streaming data</CardDescription>
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
                <p className="mt-1 font-heading text-xl font-medium">92.4%</p>
                <Progress value={92} className="mt-2 h-1" />
              </div>
              <div className="rounded-2xl bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">Headway adherence</p>
                <p className="mt-1 font-heading text-xl font-medium">88.1%</p>
                <Progress value={88} className="mt-2 h-1" />
              </div>
              <div className="rounded-2xl bg-muted/40 p-3">
                <p className="text-xs text-muted-foreground">Crowding index</p>
                <p className="mt-1 font-heading text-xl font-medium">0.34</p>
                <Progress value={34} className="mt-2 h-1" />
              </div>
            </div>
            <div className="rounded-2xl border border-dashed border-border/60 p-6 text-center text-sm text-muted-foreground">
              <HugeiconsIcon icon={ChartLineData01Icon} strokeWidth={1.5} className="mx-auto mb-2 size-6 opacity-50" />
              Inline sparkline will render here once a backend stream is connected.
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active alerts</CardTitle>
            <CardDescription>4 open · last updated 12s ago</CardDescription>
            <CardAction>
              <Button variant="link" size="sm">
                View all <HugeiconsIcon icon={ArrowUpRight01Icon} strokeWidth={1.5} className="size-3.5" />
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="space-y-2">
            {ALERTS.map((a) => (
              <div
                key={a.id}
                className="flex items-start gap-3 rounded-2xl border border-border/60 p-3"
              >
                <Badge className={SEV_COLOR[a.severity]}>{a.severity.toUpperCase()}</Badge>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{a.route}</p>
                  <p className="truncate text-xs text-muted-foreground">{a.text}</p>
                </div>
                <span className="text-xs text-muted-foreground">{a.id}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>On shift</CardTitle>
          <CardDescription>4 operators and dispatchers active right now</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            {CREW.map((c) => (
              <div key={c.initials} className="flex items-center gap-3 rounded-2xl bg-muted/40 p-2 pr-4">
                <Avatar className="size-8">
                  <AvatarFallback className="bg-primary text-primary-foreground">
                    {c.initials}
                  </AvatarFallback>
                </Avatar>
                <div className="leading-tight">
                  <p className="text-sm font-medium">{c.name}</p>
                  <p className="text-xs text-muted-foreground">{c.role}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
