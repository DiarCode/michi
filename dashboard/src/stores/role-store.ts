import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { UserRole } from "@/types"

export type Role = UserRole

interface RoleState {
  role: Role
  setRole: (role: Role) => void
}

const VALID_ROLES: Role[] = [
  "dispatch",
  "research",
  "planning",
  "executive",
  "depot",
  "passenger",
]

export const ROLE_LABELS: Record<Role, string> = {
  dispatch: "Dispatch",
  research: "Research",
  planning: "Planning",
  executive: "Executive",
  depot: "Depot",
  passenger: "Passenger",
}

export const useRoleStore = create<RoleState>()(
  persist(
    (set) => ({
      role: "dispatch",
      setRole: (role) => set({ role }),
    }),
    {
      name: "michi-role",
      merge: (persisted, current) => {
        const role = VALID_ROLES.includes((persisted as RoleState).role)
          ? (persisted as RoleState).role
          : current.role
        return { ...current, role }
      },
    }
  )
)