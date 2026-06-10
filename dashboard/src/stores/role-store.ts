import { create } from "zustand"
import { persist } from "zustand/middleware"

export type Role = "operator" | "analyst" | "executive"

interface RoleState {
  role: Role
  setRole: (role: Role) => void
}

export const ROLE_LABELS: Record<Role, string> = {
  operator: "Operator",
  analyst: "Analyst",
  executive: "Executive",
}

export const useRoleStore = create<RoleState>()(
  persist(
    (set) => ({
      role: "operator",
      setRole: (role) => set({ role }),
    }),
    { name: "michi-role" },
  ),
)
