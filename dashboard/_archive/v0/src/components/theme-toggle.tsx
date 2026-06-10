import * as React from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ComputerIcon, Moon02Icon, Sun01Icon } from "@hugeicons/core-free-icons"

import { useTheme } from "@/components/theme-provider"
import { cn } from "@/lib/utils"

interface ThemeToggleProps {
  className?: string
}

const CYCLE: Array<{ value: "system" | "light" | "dark"; label: string; icon: React.ComponentProps<typeof HugeiconsIcon>["icon"] }> = [
  { value: "system", label: "System", icon: ComputerIcon },
  { value: "light", label: "Light", icon: Sun01Icon },
  { value: "dark", label: "Dark", icon: Moon02Icon },
]

export function ThemeToggle({ className }: ThemeToggleProps) {
  const { theme, setTheme } = useTheme()
  const next = React.useMemo(() => {
    if (theme === "light") return "dark"
    if (theme === "dark") return "system"
    return "light"
  }, [theme])

  const current = CYCLE.find((c) => c.value === theme) ?? CYCLE[0]
  const Icon = current.icon

  return (
    <button
      type="button"
      aria-label={`Theme: ${current.label}. Click to switch.`}
      title={`${current.label} (press d)`}
      onClick={() => setTheme(next)}
      className={cn(
        "inline-flex size-9 items-center justify-center rounded-2xl border border-border bg-background text-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-ring/30",
        className,
      )}
    >
      <HugeiconsIcon icon={Icon} strokeWidth={2} className="size-4" />
    </button>
  )
}
