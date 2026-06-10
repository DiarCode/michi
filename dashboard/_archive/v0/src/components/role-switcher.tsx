import { HugeiconsIcon } from "@hugeicons/react"
import { UserCircleIcon } from "@hugeicons/core-free-icons"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ROLE_DESCRIPTIONS, ROLE_LABELS, useRoleStore } from "@/stores/role-store"
import type { UserRole } from "@/types"

const ROLES: UserRole[] = ["dispatch", "research", "planning", "executive", "superadmin"]

export function RoleSwitcher() {
  const { role, setRole } = useRoleStore()
  return (
    <Select value={role} onValueChange={(v) => setRole(v as UserRole)}>
      <SelectTrigger className="w-[180px] rounded-2xl">
        <HugeiconsIcon icon={UserCircleIcon} strokeWidth={2} className="size-4 text-muted-foreground" />
        <SelectValue placeholder="Select role" />
      </SelectTrigger>
      <SelectContent>
        {ROLES.map((r) => (
          <SelectItem key={r} value={r}>
            <div className="flex flex-col">
              <span className="font-medium">{ROLE_LABELS[r]}</span>
              <span className="text-xs text-muted-foreground">{ROLE_DESCRIPTIONS[r]}</span>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
