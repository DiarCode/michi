import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { HugeiconsIcon } from "@hugeicons/react"
import { ChartIcon, TrendingUp } from "@hugeicons/core-free-icons"

interface ForecastChartProps {
  title?: string
  data: { timestamp: string; predicted: number; confidence: number }[]
  loading?: boolean
}

export function ForecastChart({ title = "12h Forecast", data, loading }: ForecastChartProps) {
  const max = Math.max(1, ...data.map((d) => d.predicted))
  const min = 0
  const width = 320
  const height = 96
  const stepX = data.length > 1 ? width / (data.length - 1) : width

  const points = data.map((d, i) => {
    const x = i * stepX
    const y = height - ((d.predicted - min) / (max - min)) * height
    return { x, y, ...d }
  })

  const linePath = points
    .map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`))
    .join(" ")
  const fillPath = `${linePath} L ${width} ${height} L 0 ${height} Z`

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardDescription>{title}</CardDescription>
            <CardTitle className="text-lg">Predicted Ridership</CardTitle>
          </div>
          <span className="grid size-8 place-items-center rounded-2xl bg-muted text-foreground">
            <HugeiconsIcon icon={ChartIcon} strokeWidth={2} className="size-4" />
          </span>
        </div>
      </CardHeader>
      <CardContent>
        {loading || data.length === 0 ? (
          <div className="grid h-24 place-items-center text-sm text-muted-foreground">Loading forecast…</div>
        ) : (
          <div className="space-y-3">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="h-24 w-full"
              preserveAspectRatio="none"
            >
              <defs>
                <linearGradient id="forecast-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--chart-1)" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="var(--chart-1)" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path d={fillPath} fill="url(#forecast-fill)" />
              <path d={linePath} fill="none" stroke="var(--chart-1)" strokeWidth="2" />
            </svg>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{data[0]?.timestamp?.slice(11, 16) ?? "—"}</span>
              <span className="inline-flex items-center gap-1 text-chart-2">
                <HugeiconsIcon icon={TrendingUp} strokeWidth={2} className="size-3" />
                peak {Math.round(max)}
              </span>
              <span>{data[data.length - 1]?.timestamp?.slice(11, 16) ?? "—"}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
