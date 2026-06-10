import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  DashboardCircleIcon,
  MapPinIcon,
  Alert01Icon,
  ChartIcon,
  Analytics01Icon,
  Settings01Icon,
  PlayIcon,
  ArrowRight01Icon,
} from "@hugeicons/core-free-icons"

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"

interface CommandPaletteProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const NAV = [
  { label: "Command Center", href: "/", icon: DashboardCircleIcon, shortcut: "G C" },
  { label: "Live Map", href: "/map", icon: MapPinIcon, shortcut: "G M" },
  { label: "Alerts", href: "/alerts", icon: Alert01Icon, shortcut: "G A" },
  { label: "Simulation", href: "/simulation", icon: PlayIcon, shortcut: "G S" },
  { label: "Forecast", href: "/forecast", icon: ChartIcon, shortcut: "G F" },
  { label: "Executive", href: "/executive", icon: Analytics01Icon, shortcut: "G E" },
  { label: "Settings", href: "/settings", icon: Settings01Icon, shortcut: "G ," },
] as const

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const navigate = useNavigate()
  const [query, setQuery] = useState("")

  useEffect(() => {
    if (!open) setQuery("")
  }, [open])

  function go(href: string) {
    navigate(href)
    onOpenChange(false)
  }

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        value={query}
        onValueChange={setQuery}
        placeholder="Search routes, stations, alerts…"
      />
      <CommandList>
        <CommandEmpty>No matches for &ldquo;{query}&rdquo;.</CommandEmpty>
        <CommandGroup heading="Navigate">
          {NAV.map((item) => (
            <CommandItem key={item.href} value={item.label} onSelect={() => go(item.href)}>
              <HugeiconsIcon icon={item.icon} strokeWidth={1.5} />
              <span>{item.label}</span>
              <HugeiconsIcon
                icon={ArrowRight01Icon}
                strokeWidth={1.5}
                className="ml-auto size-3.5 text-muted-foreground"
              />
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading="Quick actions">
          <CommandItem value="toggle-theme" onSelect={() => document.documentElement.classList.toggle("dark")}>
            Toggle dark mode
          </CommandItem>
          <CommandItem value="reset-sim" onSelect={() => go("/simulation")}>
            Reset simulation
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
