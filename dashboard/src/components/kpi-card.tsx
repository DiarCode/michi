import * as React from "react"
import { cn } from "@/lib/utils"

export function KpiCard({
  label,
  value,
  delta,
  hint,
  icon,
  className,
}: {
  label: string
  value: string
  delta?: { value: string; positive?: boolean }
  hint?: string
  icon?: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {icon}
        <span className="tracking-widest uppercase">{label}</span>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="font-heading text-2xl font-medium">{value}</span>
        {delta && (
          <span
            className={cn(
              "text-xs",
              delta.positive
                ? "text-emerald-600 dark:text-emerald-400"
                : "text-rose-600 dark:text-rose-400"
            )}
          >
            {delta.value}
          </span>
        )}
      </div>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}
