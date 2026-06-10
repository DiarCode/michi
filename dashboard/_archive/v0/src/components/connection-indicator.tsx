import { HugeiconsIcon } from "@hugeicons/react"
import { CheckmarkCircle01Icon, Alert02Icon, Loading03Icon } from "@hugeicons/core-free-icons"

import { useConnectionStore } from "@/stores/connection-store"
import { cn } from "@/lib/utils"

interface ConnectionIndicatorProps {
  className?: string
}

export function ConnectionIndicator({ className }: ConnectionIndicatorProps) {
  const online = useConnectionStore((s) => s.online)
  const Icon = online ? CheckmarkCircle01Icon : Alert02Icon
  const label = online ? "Live" : "Offline"
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-2xl border border-border bg-card px-3 py-1.5 text-xs font-medium",
        className,
      )}
    >
      <HugeiconsIcon
        icon={online ? Loading03Icon : Icon}
        strokeWidth={2}
        className={cn("size-3.5", online ? "animate-spin text-chart-1" : "text-destructive")}
      />
      <HugeiconsIcon icon={Icon} strokeWidth={2} className="-ml-5 size-3.5 opacity-0" />
      <span className="tabular-nums">{label}</span>
    </div>
  )
}
