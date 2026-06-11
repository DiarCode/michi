import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { fetchPassengerCrowding } from "@/lib/api"
import type { PassengerCrowding } from "@/types"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { SectionHeader } from "@/components/section-header"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  UserGroupIcon,
  Search01Icon,
  ArrowReloadHorizontalIcon,
  ArrowDown01Icon,
  ArrowUp01Icon,
  FilterIcon,
  Clock01Icon,
} from "@hugeicons/core-free-icons"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const CROWDING_META: Record<
  string,
  { label: string; color: string; bg: string; ring: string }
> = {
  low: {
    label: "Low",
    color: "text-emerald-700 dark:text-emerald-300",
    bg: "bg-emerald-500/10",
    ring: "ring-emerald-500/20",
  },
  medium: {
    label: "Medium",
    color: "text-amber-700 dark:text-amber-300",
    bg: "bg-amber-500/10",
    ring: "ring-amber-500/20",
  },
  high: {
    label: "High",
    color: "text-rose-700 dark:text-rose-300",
    bg: "bg-rose-500/10",
    ring: "ring-rose-500/20",
  },
  very_high: {
    label: "Very High",
    color: "text-red-700 dark:text-red-300",
    bg: "bg-red-500/10",
    ring: "ring-red-500/20",
  },
}

const CROWDING_BAR_COLORS: Record<string, string> = {
  low: "#10b981",
  medium: "#f59e0b",
  high: "#f43f5e",
  very_high: "#ef4444",
}

const CROWDING_LEVELS = ["low", "medium", "high", "very_high"] as const

/* ------------------------------------------------------------------ */
/*  Sub-components                                                      */
/* ------------------------------------------------------------------ */

function CrowdingBadge({ level }: { level: string }) {
  const meta = CROWDING_META[level] ?? CROWDING_META.low
  return (
    <Badge className={`${meta.bg} ${meta.color} ring-1 ${meta.ring}`}>
      {meta.label}
    </Badge>
  )
}

function Sparkline({
  predictions,
}: {
  predictions: {
    horizon_minutes: number
    predicted: number
    confidence: number
    level: string
  }[]
}) {
  const sorted = [...predictions].sort(
    (a, b) => a.horizon_minutes - b.horizon_minutes
  )
  const maxVal = Math.max(...sorted.map((p) => p.predicted), 1)

  return (
    <div className="flex h-8 items-end gap-1">
      {sorted.map((p) => {
        const meta = CROWDING_META[p.level] ?? CROWDING_META.low
        const height = Math.max(4, (p.predicted / maxVal) * 100)
        return (
          <div
            key={p.horizon_minutes}
            className="group relative flex flex-col items-center"
          >
            <div
              className={`w-3 rounded-sm ${meta.bg} transition-all`}
              style={{ height: `${height}%`, minHeight: 4 }}
            />
            <span className="pointer-events-none absolute -top-6 left-1/2 -translate-x-1/2 rounded bg-popover px-1.5 py-0.5 text-[10px] whitespace-nowrap text-popover-foreground opacity-0 shadow-md ring-1 ring-foreground/5 transition-opacity group-hover:opacity-100">
              {p.predicted} · {Math.round(p.confidence * 100)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

function HeroStatCard({
  label,
  value,
  icon,
  accent,
}: {
  label: string
  value: React.ReactNode
  icon: React.ReactNode
  accent?: string
}) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-border/60 bg-card p-4">
      <div
        className={`flex size-9 items-center justify-center rounded-xl ${accent ?? "bg-primary/10 text-primary"}`}
      >
        {icon}
      </div>
      <div>
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="font-heading text-xl font-medium">{value}</p>
      </div>
    </div>
  )
}

function CrowdingCard({
  station,
  isExpanded,
  onToggle,
}: {
  station: PassengerCrowding["stations"][number]
  isExpanded: boolean
  onToggle: () => void
}) {
  const avgConfidence =
    station.predictions.length > 0
      ? station.predictions.reduce((s, p) => s + p.confidence, 0) /
        station.predictions.length
      : 0

  return (
    <Card
      className={`cursor-pointer transition-shadow hover:shadow-lg ${isExpanded ? "ring-1 ring-primary/30" : ""}`}
      onClick={onToggle}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <CardTitle className="truncate text-sm">{station.name}</CardTitle>
            {station.district && (
              <p className="mt-0.5 text-xs text-muted-foreground">
                {station.district}
              </p>
            )}
          </div>
          <CrowdingBadge level={station.current_crowding} />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Forecast</span>
          <span className="text-xs text-muted-foreground">
            Conf: {Math.round(avgConfidence * 100)}%
          </span>
        </div>
        <Sparkline predictions={station.predictions} />
        <div className="flex gap-1">
          {station.predictions
            .sort((a, b) => a.horizon_minutes - b.horizon_minutes)
            .map((p) => (
              <span
                key={p.horizon_minutes}
                className="text-[10px] text-muted-foreground"
              >
                {p.horizon_minutes}m
              </span>
            ))}
        </div>

        {isExpanded && (
          <div className="mt-2 space-y-2 border-t border-border/60 pt-3">
            <p className="text-xs font-medium text-foreground">
              Detailed Predictions
            </p>
            {station.predictions
              .sort((a, b) => a.horizon_minutes - b.horizon_minutes)
              .map((p) => (
                <div
                  key={p.horizon_minutes}
                  className="flex items-center justify-between rounded-xl bg-muted/40 px-3 py-2"
                >
                  <span className="text-xs text-muted-foreground">
                    +{p.horizon_minutes} min
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">{p.predicted}</span>
                    <CrowdingBadge level={p.level} />
                    <span className="text-xs text-muted-foreground">
                      {Math.round(p.confidence * 100)}%
                    </span>
                  </div>
                </div>
              ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/*  Loading skeleton                                                    */
/* ------------------------------------------------------------------ */

function PageSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-20" />
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <Skeleton className="h-9 w-56" />
        <Skeleton className="h-9 w-36" />
        <Skeleton className="h-9 w-36" />
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-48" />
        ))}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main page                                                           */
/* ------------------------------------------------------------------ */

export function CrowdingPage() {
  const [search, setSearch] = useState("")
  const [districtFilter, setDistrictFilter] = useState<string>("all")
  const [levelFilter, setLevelFilter] = useState<string>("all")
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const { data, isLoading, isError, error, refetch, dataUpdatedAt } = useQuery({
    queryKey: ["passenger-crowding"],
    queryFn: fetchPassengerCrowding,
    refetchInterval: 60_000,
  })

  const stations = data?.stations ?? []

  const districts = useMemo(
    () =>
      [...new Set(stations.map((s) => s.district).filter(Boolean))] as string[],
    [stations]
  )

  const filtered = useMemo(() => {
    let list = stations
    if (districtFilter !== "all") {
      list = list.filter((s) => s.district === districtFilter)
    }
    if (levelFilter !== "all") {
      list = list.filter((s) => s.current_crowding === levelFilter)
    }
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter(
        (s) =>
          s.name.toLowerCase().includes(q) ||
          s.district?.toLowerCase().includes(q)
      )
    }
    return list
  }, [stations, districtFilter, levelFilter, search])

  const heroStats = useMemo(() => {
    const highCount = stations.filter(
      (s) => s.current_crowding === "high" || s.current_crowding === "very_high"
    ).length
    const levelCounts = CROWDING_LEVELS.map((level) => ({
      level,
      count: stations.filter((s) => s.current_crowding === level).length,
    }))
    return { total: stations.length, highCount, levelCounts }
  }, [stations])

  const lastUpdated = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—"

  if (isLoading) {
    return (
      <div className="space-y-4">
        <SectionHeader
          eyebrow="Passenger"
          title="Crowding"
          description="Real-time station crowding levels with predictive forecasts."
        />
        <PageSkeleton />
      </div>
    )
  }

  if (isError) {
    return (
      <div className="space-y-4">
        <SectionHeader
          eyebrow="Passenger"
          title="Crowding"
          description="Real-time station crowding levels with predictive forecasts."
        />
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-sm text-destructive">
              Failed to load crowding data
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {String(error)}
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-3"
              onClick={() => refetch()}
            >
              <HugeiconsIcon
                icon={ArrowReloadHorizontalIcon}
                strokeWidth={1.5}
                className="size-3.5"
              />
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
        eyebrow="Passenger"
        title="Crowding"
        description="Real-time station crowding levels with predictive forecasts."
        actions={
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <HugeiconsIcon
              icon={ArrowReloadHorizontalIcon}
              strokeWidth={1.5}
              className="size-3.5"
            />
            Refresh
          </Button>
        }
      />

      {/* Hero stats */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <HeroStatCard
          label="Total Stations"
          value={heroStats.total}
          icon={
            <HugeiconsIcon
              icon={UserGroupIcon}
              strokeWidth={1.5}
              className="size-4"
            />
          }
        />
        <HeroStatCard
          label="High Crowding"
          value={heroStats.highCount}
          icon={
            <HugeiconsIcon
              icon={ArrowUp01Icon}
              strokeWidth={1.5}
              className="size-4"
            />
          }
          accent="bg-rose-500/10 text-rose-600 dark:text-rose-400"
        />
        <HeroStatCard
          label="Normal Stations"
          value={heroStats.total - heroStats.highCount}
          icon={
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              strokeWidth={1.5}
              className="size-4"
            />
          }
          accent="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
        />
        <HeroStatCard
          label="Last Updated"
          value={lastUpdated}
          icon={
            <HugeiconsIcon
              icon={Clock01Icon}
              strokeWidth={1.5}
              className="size-4"
            />
          }
          accent="bg-muted text-muted-foreground"
        />
      </div>

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 sm:max-w-56">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={1.5}
            className="absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            placeholder="Search stations…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-8"
          />
        </div>
        <Select value={districtFilter} onValueChange={setDistrictFilter}>
          <SelectTrigger className="w-36">
            <HugeiconsIcon
              icon={FilterIcon}
              strokeWidth={1.5}
              className="mr-1.5 size-3.5 text-muted-foreground"
            />
            <SelectValue placeholder="District" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Districts</SelectItem>
            {districts.map((d) => (
              <SelectItem key={d} value={d}>
                {d}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={levelFilter} onValueChange={setLevelFilter}>
          <SelectTrigger className="w-36">
            <SelectValue placeholder="Crowding" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Levels</SelectItem>
            {CROWDING_LEVELS.map((l) => (
              <SelectItem key={l} value={l}>
                {CROWDING_META[l].label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Summary bar chart + grid */}
      <div className="grid gap-4 lg:grid-cols-[20rem_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Distribution</CardTitle>
            <CardDescription>Stations by crowding level</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="aspect-[4/3]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={heroStats.levelCounts} layout="vertical">
                  <XAxis type="number" hide />
                  <YAxis
                    dataKey="level"
                    type="category"
                    tickFormatter={(v: string) => CROWDING_META[v]?.label ?? v}
                    width={80}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip
                    formatter={(value) => [String(value ?? 0), "Stations"]}
                    labelFormatter={(label) => CROWDING_META[String(label ?? "")]?.label ?? String(label ?? "")}
                  />
                  <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={24}>
                    {heroStats.levelCounts.map((entry) => (
                      <Cell
                        key={entry.level}
                        fill={CROWDING_BAR_COLORS[entry.level] ?? "#94a3b8"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {filtered.map((station) => (
            <CrowdingCard
              key={station.station_id}
              station={station}
              isExpanded={expandedId === station.station_id}
              onToggle={() =>
                setExpandedId((prev) =>
                  prev === station.station_id ? null : station.station_id
                )
              }
            />
          ))}
          {filtered.length === 0 && (
            <div className="col-span-full rounded-2xl border border-dashed border-border/60 bg-muted/30 py-12 text-center">
              <HugeiconsIcon
                icon={UserGroupIcon}
                strokeWidth={1.5}
                className="mx-auto size-8 text-muted-foreground/50"
              />
              <p className="mt-2 text-sm text-muted-foreground">
                No stations match your filters
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
