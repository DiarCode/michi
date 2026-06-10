import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDownRightIcon, ArrowUpRightIcon, MinusSignIcon } from "@hugeicons/core-free-icons"

import { cn, classifyDelta, formatNumber } from "@/lib/utils"

interface KpiCardProps {
  title: string
  value: number | null
  unit?: string
  delta?: number
  isCurrency?: boolean
  isPercent?: boolean
  icon?: React.ComponentProps<typeof HugeiconsIcon>["icon"]
  description?: string
  loading?: boolean
}

export function KpiCard({
  title,
  value,
  unit,
  delta,
  isCurrency,
  isPercent,
  icon: Icon,
  description,
  loading,
}: KpiCardProps) {
  const trend = delta !== undefined ? classifyDelta(delta) : "flat"
  const TrendIcon =
    trend === "up" ? ArrowUpRightIcon : trend === "down" ? ArrowDownRightIcon : MinusSignIcon
  const trendColor =
    trend === "up"
      ? "text-chart-2"
      : trend === "down"
        ? "text-destructive"
        : "text-muted-foreground"

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          {Icon && (
            <span className="grid size-8 place-items-center rounded-2xl bg-muted text-foreground">
              <HugeiconsIcon icon={Icon} strokeWidth={2} className="size-4" />
            </span>
          )}
          <CardDescription>{title}</CardDescription>
        </div>
        <CardTitle className="text-2xl font-semibold tabular-nums">
          {loading || value === null || value === undefined ? (
            <span className="text-muted-foreground">—</span>
          ) : (
            <>
              {isCurrency ? "$" : ""}
              {formatNumber(value, { maximumFractionDigits: 2 })}
              {isPercent ? "%" : ""}
              {unit ? <span className="ml-1 text-base font-normal text-muted-foreground">{unit}</span> : null}
            </>
          )}
        </CardTitle>
      </CardHeader>
      {(delta !== undefined || description) && (
        <CardContent>
          <div className="flex items-center gap-2 text-xs">
            {delta !== undefined && (
              <span className={cn("inline-flex items-center gap-1 font-medium tabular-nums", trendColor)}>
                <HugeiconsIcon icon={TrendIcon} strokeWidth={2} className="size-3" />
                {delta > 0 ? "+" : ""}
                {delta.toFixed(1)}
                {isPercent ? "%" : ""}
              </span>
            )}
            {description && <span className="text-muted-foreground">{description}</span>}
          </div>
        </CardContent>
      )}
    </Card>
  )
}
