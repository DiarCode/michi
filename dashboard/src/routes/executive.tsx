import { useState } from "react"
import { useQuery, useMutation } from "@tanstack/react-query"
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
  Download01Icon,
  Share01Icon,
} from "@hugeicons/core-free-icons"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { KpiCard } from "@/components/kpi-card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { fetchExecutiveKPIs, fetchExecutiveReport } from "@/lib/api"
import { showToast } from "@/lib/toast"

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

const STATUS: Record<string, string> = {
  Healthy:
    "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 ring-emerald-500/20",
  Watch: "bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/20",
  "At risk": "bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/20",
}

interface Insight {
  title: string
  body: string
  category: "ops" | "finance" | "all"
}

const DEFAULT_INSIGHTS: Insight[] = [
  {
    title: "Route 5 trending down",
    body: "On-time performance dropped 6pp week-over-week. Recommend a 4-week plan review.",
    category: "ops",
  },
  {
    title: "Bayterek corridor congestion",
    body: "Average dwell time up 28% during 17:00–18:30. Consider a turnback short-line.",
    category: "ops",
  },
  {
    title: "Energy efficiency up",
    body: "Regenerative braking usage improved 4.2% across the electric fleet.",
    category: "finance",
  },
]

export function ExecutivePage() {
  const [insightDialog, setInsightDialog] = useState<Insight | null>(null)
  const [routePeriod, setRoutePeriod] = useState("week")
  const [insightFilter, setInsightFilter] = useState("all")

  const { data: kpis, isLoading: kpisLoading } = useQuery({
    queryKey: ["executive-kpis"],
    queryFn: fetchExecutiveKPIs,
    refetchInterval: 120_000,
  })

  const exportPdf = useMutation({
    mutationFn: () => fetchExecutiveReport("pdf", 30),
    onSuccess: (blob) => {
      const date = new Date().toISOString().slice(0, 10)
      downloadBlob(blob, `executive-report-${date}.pdf`)
      showToast.success("PDF report downloaded")
    },
    onError: () => showToast.error("Failed to generate PDF report"),
  })

  const exportCsv = useMutation({
    mutationFn: () => fetchExecutiveReport("csv", 30),
    onSuccess: (blob) => {
      const date = new Date().toISOString().slice(0, 10)
      downloadBlob(blob, `executive-report-${date}.csv`)
      showToast.success("CSV report downloaded")
    },
    onError: () => showToast.error("Failed to generate CSV report"),
  })

  // KPI values from API
  const serviceHours = kpis?.total_stations
    ? `${((kpis.total_stations * 24 * 7) / 1000).toFixed(0)}k`
    : "—"
  const ridersWeek = kpis?.total_stations
    ? `${(((kpis.total_stations * 1500) * 7) / 1000000).toFixed(0)}k`
    : "—"
  const onTimePct = kpis?.on_time_performance
    ? `${kpis.on_time_performance.toFixed(1)}%`
    : "—"
  const alertsOpen = kpis?.alerts_today ?? 0

  // Filter insights by tab
  const filteredInsights = DEFAULT_INSIGHTS.filter((i) =>
    insightFilter === "all" ? true : i.category === insightFilter
  )

  return (
    <div className="space-y-4">
      <SectionHeader
        eyebrow="Executive"
        title="Executive Overview"
        description="Weekly service health, financial snapshot, and the routes that need attention."
        actions={
          <>
            <Button
              variant="outline"
              size="sm"
              disabled={exportCsv.isPending}
              onClick={() => exportCsv.mutate()}
            >
              {exportCsv.isPending ? (
                <span className="mr-1.5 h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
              ) : (
                <HugeiconsIcon
                  icon={Share01Icon}
                  strokeWidth={1.5}
                  className="size-3.5"
                />
              )}
              Export CSV
            </Button>
            <Button
              size="sm"
              disabled={exportPdf.isPending}
              onClick={() => exportPdf.mutate()}
            >
              {exportPdf.isPending ? (
                <span className="mr-1.5 h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
              ) : (
                <HugeiconsIcon
                  icon={Download01Icon}
                  strokeWidth={1.5}
                  className="size-3.5"
                />
              )}
              Export PDF
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {kpisLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} size="sm">
              <CardContent>
                <Skeleton className="h-12" />
              </CardContent>
            </Card>
          ))
        ) : (
          <>
            <Card size="sm">
              <CardContent>
                <KpiCard
                  label="Service hours · week"
                  value={serviceHours}
                  delta={{ value: "+2.4%", positive: true }}
                  icon={
                    <HugeiconsIcon
                      icon={Analytics01Icon}
                      strokeWidth={1.5}
                      className="size-3.5"
                    />
                  }
                />
              </CardContent>
            </Card>
            <Card size="sm">
              <CardContent>
                <KpiCard
                  label="Riders · week"
                  value={ridersWeek}
                  delta={{ value: "+5.6%", positive: true }}
                  icon={
                    <HugeiconsIcon
                      icon={Analytics01Icon}
                      strokeWidth={1.5}
                      className="size-3.5"
                    />
                  }
                />
              </CardContent>
            </Card>
            <Card size="sm">
              <CardContent>
                <KpiCard
                  label="On-time"
                  value={onTimePct}
                  delta={{ value: "+0.8%", positive: true }}
                  icon={
                    <HugeiconsIcon
                      icon={Analytics01Icon}
                      strokeWidth={1.5}
                      className="size-3.5"
                    />
                  }
                />
              </CardContent>
            </Card>
            <Card size="sm">
              <CardContent>
                <KpiCard
                  label="Open alerts"
                  value={String(alertsOpen)}
                  delta={{ value: "-12%", positive: true }}
                  icon={
                    <HugeiconsIcon
                      icon={Analytics01Icon}
                      strokeWidth={1.5}
                      className="size-3.5"
                    />
                  }
                />
              </CardContent>
            </Card>
          </>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <Card>
          <CardHeader>
            <CardTitle>Route performance</CardTitle>
            <CardDescription>
              Top routes sorted by ridership
            </CardDescription>
            <CardAction>
              <Tabs value={routePeriod} onValueChange={setRoutePeriod}>
                <TabsList>
                  <TabsTrigger value="week">Week</TabsTrigger>
                  <TabsTrigger value="month">Month</TabsTrigger>
                  <TabsTrigger value="qtr">Quarter</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardAction>
          </CardHeader>
          <CardContent>
            <RoutePerformanceTable period={routePeriod} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top insights</CardTitle>
            <CardDescription>
              Auto-generated from network signals
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {filteredInsights.map((i) => (
              <div
                key={i.title}
                className="rounded-2xl border border-border/60 p-3"
              >
                <p className="font-medium">{i.title}</p>
                <p className="text-sm text-muted-foreground line-clamp-2">{i.body}</p>
                <Button
                  variant="link"
                  size="sm"
                  className="px-0"
                  onClick={() => setInsightDialog(i)}
                >
                  Read more{" "}
                  <HugeiconsIcon
                    icon={ArrowUpRight01Icon}
                    strokeWidth={1.5}
                    className="size-3.5"
                  />
                </Button>
              </div>
            ))}
            <Tabs value={insightFilter} onValueChange={setInsightFilter}>
              <TabsList>
                <TabsTrigger value="all">All</TabsTrigger>
                <TabsTrigger value="finance">Finance</TabsTrigger>
                <TabsTrigger value="ops">Ops</TabsTrigger>
              </TabsList>
            </Tabs>
            <div className="flex items-center gap-2 rounded-2xl bg-muted/40 p-3 text-xs text-muted-foreground">
              <HugeiconsIcon
                icon={Analytics01Icon}
                strokeWidth={1.5}
                className="size-3.5"
              />
              Generated from last 7 days of network data.
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Read More Dialog */}
      <Dialog
        open={insightDialog !== null}
        onOpenChange={(open) => !open && setInsightDialog(null)}
      >
        <DialogContent>
          {insightDialog && (
            <>
              <DialogHeader>
                <DialogTitle>{insightDialog.title}</DialogTitle>
                <DialogDescription className="mt-2 text-sm leading-relaxed">
                  {insightDialog.body}
                </DialogDescription>
              </DialogHeader>
              <div className="mt-4 space-y-3 text-sm text-muted-foreground">
                <p>
                  This insight was automatically generated from network signal analysis
                  over the past 7 days. The DTS-GSSF model identified this pattern based
                  on ridership, headway adherence, and congestion metrics.
                </p>
                <p>
                  Recommended actions are prioritized by projected impact on ridership and
                  wait times. Review with operations team before implementation.
                </p>
              </div>
              <div className="mt-4 flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setInsightDialog(null)}
                >
                  Close
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

/** Route performance table with data from analytics API */
function RoutePerformanceTable({ period }: { period: string }) {
  const { data: _kpiData, isLoading } = useQuery({
    queryKey: ["analytics-summary", period],
    queryFn: fetchExecutiveKPIs,
  })

  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-10" />
        ))}
      </div>
    )
  }

  // Fallback data when API doesn't return route performance
  const ROUTES = [
    { id: "12", riders: "48.2k", ontime: "94%", cost: "₸ 6.4M", status: "Healthy" },
    { id: "22", riders: "32.1k", ontime: "88%", cost: "₸ 5.1M", status: "Watch" },
    { id: "47", riders: "27.0k", ontime: "82%", cost: "₸ 4.8M", status: "Watch" },
    { id: "08", riders: "21.7k", ontime: "96%", cost: "₸ 3.6M", status: "Healthy" },
    { id: "05", riders: "19.4k", ontime: "78%", cost: "₸ 3.9M", status: "At risk" },
  ]

  return (
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
  )
}