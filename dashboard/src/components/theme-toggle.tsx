import { HugeiconsIcon } from "@hugeicons/react"
import { MoonIcon, SunIcon, ComputerIcon } from "@hugeicons/core-free-icons"
import { useTheme } from "@/components/theme-provider"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export function ThemeToggle() {
  const { setTheme } = useTheme()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon-xs" aria-label="Toggle theme">
          <HugeiconsIcon
            icon={SunIcon}
            strokeWidth={1.5}
            className="dark:hidden"
          />
          <HugeiconsIcon
            icon={MoonIcon}
            strokeWidth={1.5}
            className="hidden dark:block"
          />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}>
          <HugeiconsIcon icon={SunIcon} strokeWidth={1.5} /> Light
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}>
          <HugeiconsIcon icon={MoonIcon} strokeWidth={1.5} /> Dark
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}>
          <HugeiconsIcon icon={ComputerIcon} strokeWidth={1.5} /> System
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
