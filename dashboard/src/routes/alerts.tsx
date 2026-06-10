import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
} from "@/components/ui/card"
import { SectionHeader } from "@/components/section-header"
import { HugeiconsIcon } from "@hugeicons/react"
import { Alert02Icon, CheckmarkCircle01Icon, FilterIcon } from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"

const ALERTS = [
  { id: "A-201", sev: "high", route: "12", title: "Heavy congestion · Bayterek", opened: "12s ago" },
  { id: "A-202", sev: "med", route: "47", title: "Vehicle substitution required", opened: "2m ago" },
  { id: "A-203", sev: "low", route: "08", title: "Off-schedule stop request", opened: "9m ago" },
  { id: "A-204", sev: "med", route: "22", title: "Door sensor intermittent", opened: "14m ago" },
  { id: "A-205", sev: "high", route: "05", title: "Operator absent — shift gap", opened: "22m ago" },
]

const SEV: Record<string, string> = {
  high: "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
  med: "bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/20",
  low: "bg-zinc-500/10 text-zinc-700 dark:text-zinc-300 ring-zinc-500/20",
}

export function AlertsPage() {
  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="Operations"
        title="Alerts"
        description="Triage active and historical alerts. Filter by severity, route, and time window."
        actions={
          <>
            <Button variant="outline" size="sm">
              <HugeiconsIcon icon={FilterIcon} strokeWidth={1.5} className="size-3.5" />
              Filters
            </Button>
            <Button size="sm">
              <HugeiconsIcon icon={CheckmarkCircle01Icon} strokeWidth={1.5} className="size-3.5" />
              Acknowledge all
            </Button>
          </>
        }
      />

      <Card>
        <CardHeader>
          <Tabs defaultValue="open">
            <TabsList>
              <TabsTrigger value="open">Open · 5</TabsTrigger>
              <TabsTrigger value="ack">Acknowledged · 8</TabsTrigger>
              <TabsTrigger value="closed">Closed · 41</TabsTrigger>
            </TabsList>
          </Tabs>
          <CardAction>
            <Input className="w-56" placeholder="Search alerts…" />
          </CardAction>
        </CardHeader>
        <CardContent className="space-y-2">
          {ALERTS.map((a) => (
            <div
              key={a.id}
              className="flex flex-wrap items-center gap-3 rounded-2xl border border-border/60 p-3"
            >
              <HugeiconsIcon
                icon={Alert02Icon}
                strokeWidth={1.5}
                className="size-4 text-muted-foreground"
              />
              <Badge className={SEV[a.sev]}>{a.sev.toUpperCase()}</Badge>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">{a.title}</p>
                <p className="text-xs text-muted-foreground">
                  Route {a.route} · {a.id} · opened {a.opened}
                </p>
              </div>
              <Button variant="outline" size="sm">
                Open
              </Button>
              <Button size="sm">Resolve</Button>
            </div>
          ))}
          {ALERTS.length === 0 && (
            <Empty>
              <EmptyHeader>
                <EmptyTitle>No open alerts</EmptyTitle>
                <EmptyDescription>Everything is running smoothly across the network.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
