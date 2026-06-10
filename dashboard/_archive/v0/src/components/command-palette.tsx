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
  CommandShortcut,
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
]

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const navigate = useNavigate()
  const [tick, setTick] = useState(0)

  useEffect(() => {
    if (open) setTick((t) => t + 1)
  }, [open])

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange} key={tick}>
      <CommandInput placeholder="Search stations, routes, alerts…" />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        <CommandGroup heading="Navigate">
          {NAV.map((item) => {
            const Icon = item.icon
            return (
              <CommandItem
                key={item.href}
                value={item.label}
                onSelect={() => {
                  onOpenChange(false)
                  navigate(item.href)
                }}
              >
                <HugeiconsIcon icon={Icon} strokeWidth={2} />
                <span>{item.label}</span>
                <CommandShortcut>{item.shortcut}</CommandShortcut>
              </CommandItem>
            )
          })}
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading="Quick actions">
          <CommandItem
            value="open command center"
            onSelect={() => {
              onOpenChange(false)
              navigate("/")
            }}
          >
            <HugeiconsIcon icon={DashboardCircleIcon} strokeWidth={2} />
            <span>Open command center</span>
            <CommandShortcut>
              <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} />
            </CommandShortcut>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
