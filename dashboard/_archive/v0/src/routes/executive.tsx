import { useQuery } from "@tanstack/react-query"
import { HugeiconsIcon } from "@hugeicons/react"
import { Analytics01Icon, BanknoteIcon, DollarSignIcon, TargetIcon, UserCheckIcon } from "@hugeicons/core-free-icons"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { KpiCard } from "@/components/kpi-card"
import { Skeleton } from "@/components/ui/skeleton"
import { fetchExecutiveKPIs, fetchROISummary, fetchFinancialSummary } from "@/lib/api"

export function ExecutivePage() {
  const kpis = useQuery({ queryKey: ["exec-kpis"], queryFn: fetchExecutiveKPIs })
  const roi = useQuery({ queryKey: ["exec-roi"], queryFn: fetchROISummary })
  const fin = useQuery({ queryKey: ["exec-fin"], queryFn: fetchFinancialSummary })

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">Executive</p>
        <h1 className="font-heading text-3xl font-medium tracking-tight">Quarterly briefing</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Top-line KPIs, ROI from interventions and financial pulse for the Astana bus network.
        </p>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="Ridership"
          value={kpis.data?.ridership_total ?? null}
          delta={kpis.data?.ridership_delta}
          icon={TargetIcon}
          description="vs. last month"
          loading={kpis.isLoading}
        />
        <KpiCard
          title="On-time"
          value={kpis.data?.on_time ?? null}
          isPercent
          delta={kpis.data?.on_time_delta}
          icon={Analytics01Icon}
          loading={kpis.isLoading}
        />
        <KpiCard
          title="Cost per ride"
          value={kpis.data?.cost_per_ride ?? null}
          isCurrency
          delta={kpis.data?.cost_per_ride_delta}
          icon={DollarSignIcon}
          loading={kpis.isLoading}
        />
        <KpiCard
          title="Satisfaction"
          value={kpis.data?.customer_satisfaction ?? null}
          isPercent
          delta={kpis.data?.customer_satisfaction_delta}
          icon={UserCheckIcon}
          loading={kpis.isLoading}
        />
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardDescription>ROI</CardDescription>
            <CardTitle className="text-lg">Intervention return on investment</CardTitle>
          </CardHeader>
          <CardContent>
            {roi.isLoading ? (
              <Skeleton className="h-40 w-full rounded-2xl" />
            ) : (
              <div className="space-y-3">
                {Object.entries(roi.data ?? {}).slice(0, 6).map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-center justify-between rounded-2xl border border-border bg-card p-3"
                  >
                    <span className="text-sm font-medium capitalize">{k.replace(/_/g, " ")}</span>
                    <span className="font-heading text-sm tabular-nums text-chart-2">
                      {typeof v === "number" ? v.toFixed(2) : String(v ?? "—")}
                    </span>
                  </div>
                ))}
                {Object.keys(roi.data ?? {}).length === 0 && (
                  <p className="rounded-2xl border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
                    ROI data will appear here once interventions execute.
                  </p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardDescription>Financials</CardDescription>
            <CardTitle className="text-lg">Operating snapshot</CardTitle>
          </CardHeader>
          <CardContent>
            {fin.isLoading ? (
              <Skeleton className="h-40 w-full rounded-2xl" />
            ) : (
              <div className="space-y-3">
                {Object.entries(fin.data ?? {}).slice(0, 6).map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-center justify-between rounded-2xl border border-border bg-card p-3"
                  >
                    <span className="inline-flex items-center gap-2 text-sm font-medium capitalize">
                      <HugeiconsIcon icon={BanknoteIcon} strokeWidth={2} className="size-3.5 text-muted-foreground" />
                      {k.replace(/_/g, " ")}
                    </span>
                    <span className="font-heading text-sm tabular-nums">
                      {typeof v === "number" ? v.toLocaleString() : String(v ?? "—")}
                    </span>
                  </div>
                ))}
                {Object.keys(fin.data ?? {}).length === 0 && (
                  <p className="rounded-2xl border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
                    Financial summary pending.
                  </p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

export default ExecutivePage
