import { useEffect, useState } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ComputerIcon, LightbulbOff, Moon02Icon, PaintBrush01Icon, Settings02Icon, Sun01Icon, UserCircleIcon } from "@hugeicons/core-free-icons"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Label } from "@/components/ui/label"
import { useTheme } from "@/components/theme-provider"
import { ROLE_DESCRIPTIONS, ROLE_LABELS, useRoleStore } from "@/stores/role-store"
import type { UserRole } from "@/types"

const ROLES: UserRole[] = ["dispatch", "research", "planning", "executive", "superadmin"]

const THEME_OPTIONS = [
  { value: "light", label: "Light", icon: Sun01Icon },
  { value: "dark", label: "Dark", icon: Moon02Icon },
  { value: "system", label: "System", icon: ComputerIcon },
] as const

export function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const { role, setRole } = useRoleStore()
  const [density, setDensity] = useState<"comfortable" | "compact">("comfortable")
  const [notifications, setNotifications] = useState(true)
  const [autorefetch, setAutorefetch] = useState(true)

  useEffect(() => {
    const d = localStorage.getItem("michi-density")
    if (d === "compact" || d === "comfortable") setDensity(d)
  }, [])
  useEffect(() => {
    localStorage.setItem("michi-density", density)
  }, [density])

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">Settings</p>
        <h1 className="font-heading text-3xl font-medium tracking-tight">Workspace preferences</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Tweak the interface to match how you work. All settings are saved to this browser.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <span className="grid size-9 place-items-center rounded-2xl bg-muted text-foreground">
                <HugeiconsIcon icon={PaintBrush01Icon} strokeWidth={2} className="size-4" />
              </span>
              <div>
                <CardDescription>Appearance</CardDescription>
                <CardTitle className="text-lg">Theme & density</CardTitle>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="space-y-2">
              <Label className="text-xs uppercase tracking-widest text-muted-foreground">Theme</Label>
              <ToggleGroup
                value={[theme]}
                onValueChange={(v) => v[0] && setTheme(v[0] as typeof theme)}
                variant="outline"
                spacing={0}
                className="w-full"
              >
                {THEME_OPTIONS.map((opt) => {
                  const Icon = opt.icon
                  return (
                    <ToggleGroupItem key={opt.value} value={opt.value} className="flex-1">
                      <HugeiconsIcon icon={Icon} strokeWidth={2} />
                      {opt.label}
                    </ToggleGroupItem>
                  )
                })}
              </ToggleGroup>
              <p className="text-xs text-muted-foreground">Tip: press <kbd className="rounded border px-1">d</kbd> to toggle.</p>
            </div>
            <div className="space-y-2">
              <Label className="text-xs uppercase tracking-widest text-muted-foreground">Density</Label>
              <ToggleGroup
                value={[density]}
                onValueChange={(v) => v[0] && setDensity(v[0] as typeof density)}
                variant="outline"
                spacing={0}
                className="w-full"
              >
                <ToggleGroupItem value="comfortable" className="flex-1">Comfortable</ToggleGroupItem>
                <ToggleGroupItem value="compact" className="flex-1">Compact</ToggleGroupItem>
              </ToggleGroup>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <span className="grid size-9 place-items-center rounded-2xl bg-muted text-foreground">
                <HugeiconsIcon icon={UserCircleIcon} strokeWidth={2} className="size-4" />
              </span>
              <div>
                <CardDescription>Role</CardDescription>
                <CardTitle className="text-lg">What can I see?</CardTitle>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <ToggleGroup
              value={[role]}
              onValueChange={(v) => v[0] && setRole(v[0] as UserRole)}
              variant="outline"
              spacing={0}
              className="flex w-full flex-wrap"
            >
              {ROLES.map((r) => (
                <ToggleGroupItem key={r} value={r} className="flex-1">
                  {ROLE_LABELS[r]}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <p className="text-sm text-muted-foreground">{ROLE_DESCRIPTIONS[role]}</p>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center gap-2">
              <span className="grid size-9 place-items-center rounded-2xl bg-muted text-foreground">
                <HugeiconsIcon icon={Settings02Icon} strokeWidth={2} className="size-4" />
              </span>
              <div>
                <CardDescription>Behavior</CardDescription>
                <CardTitle className="text-lg">Live data & alerts</CardTitle>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2">
            <SettingRow
              icon={LightbulbOff}
              title="Push notifications"
              description="Surface critical alerts as toasts."
              value={notifications}
              onChange={setNotifications}
            />
            <SettingRow
              icon={Settings02Icon}
              title="Auto-refresh data"
              description="Keep dashboards live without reloading."
              value={autorefetch}
              onChange={setAutorefetch}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

interface SettingRowProps {
  icon: React.ComponentProps<typeof HugeiconsIcon>["icon"]
  title: string
  description: string
  value: boolean
  onChange: (v: boolean) => void
}

function SettingRow({ icon: Icon, title, description, value, onChange }: SettingRowProps) {
  return (
    <div className="flex items-center justify-between rounded-2xl border border-border bg-card p-4">
      <div className="flex items-center gap-3">
        <span className="grid size-9 place-items-center rounded-2xl bg-muted text-foreground">
          <HugeiconsIcon icon={Icon} strokeWidth={2} className="size-4" />
        </span>
        <div>
          <p className="font-medium">{title}</p>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
      </div>
      <button
        role="switch"
        aria-checked={value}
        onClick={() => onChange(!value)}
        className={
          "relative inline-flex h-6 w-11 items-center rounded-full transition-colors " +
          (value ? "bg-primary" : "bg-muted")
        }
      >
        <span
          className={
            "inline-block size-5 transform rounded-full bg-background shadow transition-transform " +
            (value ? "translate-x-5" : "translate-x-0.5")
          }
        />
      </button>
    </div>
  )
}

export default SettingsPage
