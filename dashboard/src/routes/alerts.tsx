import { useState, useMemo } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Card, CardAction, CardContent, CardHeader } from "@/components/ui/card"
import { SectionHeader } from "@/components/section-header"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Alert02Icon,
  CheckmarkCircle01Icon,
  FilterIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { fetchRichAlerts, ackAlert } from "@/lib/api"
import { showToast } from "@/lib/toast"

const SEV: Record<string, string> = {
  critical: "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
  high: "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
  warning: "bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/20",
  med: "bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/20",
  info: "bg-blue-500/10 text-blue-700 dark:text-blue-300 ring-blue-500/20",
  low: "bg-zinc-500/10 text-zinc-700 dark:text-zinc-300 ring-zinc-500/20",
}

const SEV_ORDER: Record<string, number> = {
  critical: 0,
  high: 1,
  warning: 2,
  med: 2,
  info: 3,
  low: 4,
}

function formatTimeAgo(dateStr: string): string {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return "just now"
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  return date.toLocaleDateString()
}

export function AlertsPage() {
  const queryClient = useQueryClient()
  const [tab, setTab] = useState("open")
  const [search, setSearch] = useState("")
  const [severityFilter, setSeverityFilter] = useState("all")

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["alerts-rich"],
    queryFn: fetchRichAlerts,
    refetchInterval: 30_000,
  })

  const ackMutation = useMutation({
    mutationFn: (alertId: number) => ackAlert(alertId),
    onSuccess: () => {
      showToast.success("Alert acknowledged")
      queryClient.invalidateQueries({ queryKey: ["alerts-rich"] })
    },
    onError: () => showToast.error("Failed to acknowledge alert"),
  })

  const ackAllMutation = useMutation({
    mutationFn: async () => {
      const openAlerts = alerts.filter(
        (a) => !a.acknowledged && tab === "open"
      )
      for (const a of openAlerts) {
        await ackAlert(a.id)
      }
    },
    onSuccess: () => {
      showToast.success("All alerts acknowledged")
      queryClient.invalidateQueries({ queryKey: ["alerts-rich"] })
    },
    onError: () => showToast.error("Failed to acknowledge some alerts"),
  })

  const alerts = data?.alerts ?? []

  const filtered = useMemo(() => {
    let list = [...alerts]
    // Tab filter
    if (tab === "open") list = list.filter((a) => !a.acknowledged)
    else if (tab === "ack") list = list.filter((a) => a.acknowledged)
    // Severity filter
    if (severityFilter !== "all")
      list = list.filter((a) => a.severity === severityFilter)
    // Search
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter(
        (a) =>
          a.title.toLowerCase().includes(q) ||
          a.what?.toLowerCase().includes(q) ||
          a.route_id?.toLowerCase().includes(q) ||
          a.station_id?.toLowerCase().includes(q)
      )
    }
    // Sort by severity then time
    list.sort((a, b) => {
      const sa = SEV_ORDER[a.severity] ?? 5
      const sb = SEV_ORDER[b.severity] ?? 5
      if (sa !== sb) return sa - sb
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    })
    return list
  }, [alerts, tab, severityFilter, search])

  const openCount = alerts.filter((a) => !a.acknowledged).length
  const ackCount = alerts.filter((a) => a.acknowledged).length

  if (isLoading) {
    return (
      <div className="space-y-4">
        <SectionHeader
          eyebrow="Operations"
          title="Alerts"
          description="Triage active and historical alerts."
        />
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="space-y-4">
        <SectionHeader
          eyebrow="Operations"
          title="Alerts"
          description="Triage active and historical alerts."
        />
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-sm text-destructive">Failed to load alerts</p>
            <p className="mt-1 text-xs text-muted-foreground">{String(error)}</p>
            <Button variant="outline" size="sm" className="mt-3" onClick={() => refetch()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="Operations"
        title="Alerts"
        description="Triage active and historical alerts. Filter by severity, route, and time window."
        actions={
          <>
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger className="w-32">
                <HugeiconsIcon
                  icon={FilterIcon}
                  strokeWidth={1.5}
                  className="mr-1.5 size-3.5 text-muted-foreground"
                />
                <SelectValue placeholder="Severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="info">Info</SelectItem>
              </SelectContent>
            </Select>
            <Button
              size="sm"
              onClick={() => ackAllMutation.mutate()}
              disabled={ackAllMutation.isPending || openCount === 0}
            >
              <HugeiconsIcon
                icon={CheckmarkCircle01Icon}
                strokeWidth={1.5}
                className="size-3.5"
              />
              Acknowledge all
            </Button>
          </>
        }
      />

      <Card>
        <CardHeader>
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList>
              <TabsTrigger value="open">Open · {openCount}</TabsTrigger>
              <TabsTrigger value="ack">Acknowledged · {ackCount}</TabsTrigger>
              <TabsTrigger value="all">All · {alerts.length}</TabsTrigger>
            </TabsList>
          </Tabs>
          <CardAction>
            <div className="relative">
              <HugeiconsIcon
                icon={Search01Icon}
                strokeWidth={1.5}
                className="absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground"
              />
              <Input
                className="w-56 pl-8"
                placeholder="Search alerts…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </CardAction>
        </CardHeader>
        <CardContent className="space-y-2">
          {filtered.map((a) => (
            <div
              key={a.id}
              className="flex flex-wrap items-center gap-3 rounded-2xl border border-border/60 p-3"
            >
              <HugeiconsIcon
                icon={Alert02Icon}
                strokeWidth={1.5}
                className="size-4 text-muted-foreground"
              />
              <Badge className={SEV[a.severity] ?? SEV.low}>
                {a.severity.toUpperCase()}
              </Badge>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">{a.title}</p>
                <p className="text-xs text-muted-foreground">
                  {a.route_id ? `Route ${a.route_id} · ` : ""}
                  {a.station_id ? `Station ${a.station_id} · ` : ""}
                  {formatTimeAgo(a.created_at)}
                </p>
                {a.what && (
                  <p className="mt-0.5 text-xs text-muted-foreground line-clamp-1">
                    {a.what}
                  </p>
                )}
              </div>
              {!a.acknowledged && (
                <div className="flex gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => ackMutation.mutate(a.id)}
                    disabled={ackMutation.isPending}
                  >
                    Acknowledge
                  </Button>
                </div>
              )}
              {a.acknowledged && (
                <Badge variant="secondary" className="text-xs">
                  ✓ Acknowledged
                </Badge>
              )}
            </div>
          ))}
          {filtered.length === 0 && (
            <Empty>
              <EmptyHeader>
                <EmptyTitle>No alerts match your filters</EmptyTitle>
                <EmptyDescription>
                  {tab === "open"
                    ? "Everything is running smoothly across the network."
                    : "No alerts match the current filters."}
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </CardContent>
      </Card>
    </div>
  )
}