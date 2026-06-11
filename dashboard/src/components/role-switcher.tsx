import { HugeiconsIcon } from "@hugeicons/react"
import {
  UserIcon,
  Analytics01Icon,
  DashboardCircleIcon,
} from "@hugeicons/core-free-icons"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useRoleStore, ROLE_LABELS, type Role } from "@/stores/role-store"

const ICONS: Record<Role, typeof UserIcon> = {
  dispatch: UserIcon,
  research: Analytics01Icon,
  planning: Analytics01Icon,
  executive: DashboardCircleIcon,
  depot: UserIcon,
  passenger: UserIcon,
}

export function RoleSwitcher() {
  const role = useRoleStore((s) => s.role)
  const setRole = useRoleStore((s) => s.setRole)

  return (
    <Select value={role} onValueChange={(v) => setRole(v as Role)}>
      <SelectTrigger
        size="sm"
        className="h-7 min-w-[8rem] rounded-2xl border-border/60 bg-muted/40 text-xs"
      >
        <HugeiconsIcon
          icon={ICONS[role] ?? UserIcon}
          strokeWidth={1.5}
          className="size-3.5"
        />
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {(Object.keys(ROLE_LABELS) as Role[]).map((r) => (
          <SelectItem key={r} value={r}>
            <HugeiconsIcon
              icon={ICONS[r] ?? UserIcon}
              strokeWidth={1.5}
              className="size-3.5"
            />
            {ROLE_LABELS[r]}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
