import { useQuery } from "@tanstack/react-query"
import { HugeiconsIcon } from "@hugeicons/react"
import { Alert02Icon, CheckmarkCircle01Icon, Clock01Icon, FilterIcon, MapPinIcon, RouteIcon, Search01Icon } from "@hugeicons/core-free-icons"
import { useMemo, useState } from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { fetchRichAlerts, ackAlert } from "@/lib/api"
import { formatRelativeTime } from "@/lib/utils"
import { toast } from "sonner"

type Filter = "all" | "critical" | "warning" | "info"

export function AlertsPage() {
  const [filter, setFilter] = useState<Filter>("all")
  const [query, setQuery] = useState("")
  const alerts = useQuery({
    queryKey: ["rich-alerts", "alerts-page"],
    queryFn: fetchRichAlerts,
    refetchInterval: 10_000,
  })

  const filtered = useMemo(() => {
    return (alerts.data?.alerts ?? [])
      .filter((a) => (filter === "all" ? true : a.severity === filter))
      .filter((a) => (query ? a.title.toLowerCase().includes(query.toLowerCase()) : true))
  }, [alerts.data, filter, query])

  const counts = useMemo(() => {
    const list = alerts.data?.alerts ?? []
    return {
      all: list.length,
      critical: list.filter((a) => a.severity === "critical").length,
      warning: list.filter((a) => a.severity === "warning").length,
      info: list.filter((a) => a.severity === "info").length,
    }
  }, [alerts.data])

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">Alerts</p>
        <h1 className="font-heading text-3xl font-medium tracking-tight">Operational signal feed</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Rich alerts with confidence, family and recommended actions. Acknowledge to dismiss from the feed.
        </p>
      </header>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Tabs value={filter} onValueChange={(v) => setFilter(v as Filter)}>
          <TabsList>
            <TabsTrigger value="all">All ({counts.all})</TabsTrigger>
            <TabsTrigger value="critical">Critical ({counts.critical})</TabsTrigger>
            <TabsTrigger value="warning">Warning ({counts.warning})</TabsTrigger>
            <TabsTrigger value="info">Info ({counts.info})</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="flex items-center gap-2">
          <div className="relative w-full sm:w-72">
            <HugeiconsIcon
              icon={Search01Icon}
              strokeWidth={2}
              className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search alerts…"
              className="pl-9"
            />
          </div>
          <Button variant="outline" size="icon">
            <HugeiconsIcon icon={FilterIcon} strokeWidth={2} />
          </Button>
        </div>
      </div>

      {alerts.isLoading ? (
        <div className="grid gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-32 w-full rounded-3xl" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card>
          <CardContent className="grid place-items-center gap-2 py-16 text-center">
            <span className="grid size-12 place-items-center rounded-full bg-muted text-chart-2">
              <HugeiconsIcon icon={CheckmarkCircle01Icon} strokeWidth={2} className="size-6" />
            </span>
            <p className="font-medium">Inbox zero</p>
            <p className="text-sm text-muted-foreground">No alerts matching your filter.</p>
          </CardContent>
        </Card>
      ) : (
        <ul className="grid gap-3">
          {filtered.map((a) => (
            <li key={a.id}>
              <Card>
                <CardHeader>
                  <div className="flex items-start gap-3">
                    <span
                      className={
                        a.severity === "critical"
                          ? "mt-1 grid size-9 place-items-center rounded-2xl bg-destructive/10 text-destructive"
                          : a.severity === "warning"
                            ? "mt-1 grid size-9 place-items-center rounded-2xl bg-chart-3/15 text-chart-3"
                            : "mt-1 grid size-9 place-items-center rounded-2xl bg-muted text-muted-foreground"
                      }
                    >
                      <HugeiconsIcon icon={Alert02Icon} strokeWidth={2} className="size-4" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <CardTitle className="text-base">{a.title}</CardTitle>
                        <Badge
                          variant={
                            a.severity === "critical"
                              ? "destructive"
                              : a.severity === "warning"
                                ? "default"
                                : "secondary"
                          }
                        >
                          {a.severity}
                        </Badge>
                        {a.family && (
                          <Badge variant="outline" className="rounded-xl capitalize">
                            {a.family}
                          </Badge>
                        )}
                        <span className="ml-auto inline-flex items-center gap-1 text-xs text-muted-foreground">
                          <HugeiconsIcon icon={Clock01Icon} strokeWidth={2} className="size-3" />
                          {formatRelativeTime(a.created_at)}
                        </span>
                      </div>
                      <CardDescription className="mt-1">{a.message}</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {a.why && (
                    <p className="mb-3 text-xs text-muted-foreground">
                      <span className="font-medium text-foreground">Why: </span>
                      {a.why}
                    </p>
                  )}
                  {a.recommended_actions && a.recommended_actions.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {a.recommended_actions.map((act) => (
                        <Button
                          key={act.type}
                          size="sm"
                          variant="outline"
                          className="rounded-2xl"
                          onClick={() => toast.success(`${act.label} queued`)}
                        >
                          {act.label}
                        </Button>
                      ))}
                    </div>
                  )}
                  <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                    {a.station_id && (
                      <span className="inline-flex items-center gap-1">
                        <HugeiconsIcon icon={MapPinIcon} strokeWidth={2} className="size-3" /> {a.station_id}
                      </span>
                    )}
                    {a.route_id && (
                      <span className="inline-flex items-center gap-1">
                        <HugeiconsIcon icon={RouteIcon} strokeWidth={2} className="size-3" /> {a.route_id}
                      </span>
                    )}
                    <Button
                      size="xs"
                      variant="ghost"
                      className="ml-auto rounded-xl"
                      onClick={async () => {
                        try {
                          await ackAlert(a.id)
                          toast.success(`Alert #${a.id} acknowledged`)
                          alerts.refetch()
                        } catch {
                          toast.error("Could not acknowledge")
                        }
                      }}
                    >
                      Acknowledge
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default AlertsPage
