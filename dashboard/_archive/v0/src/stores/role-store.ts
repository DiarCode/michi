import { create } from "zustand"
import type { UserRole } from "@/types"

interface RoleState {
  role: UserRole
  setRole: (role: UserRole) => void
}

const VALID_ROLES: UserRole[] = ["dispatch", "research", "planning", "executive", "superadmin"]

function readRole(): UserRole {
  if (typeof window === "undefined") return "dispatch"
  const stored = window.localStorage.getItem("michi-role")
  return VALID_ROLES.includes(stored as UserRole) ? (stored as UserRole) : "dispatch"
}

export const useRoleStore = create<RoleState>((set) => ({
  role: readRole(),
  setRole: (role) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("michi-role", role)
    }
    set({ role })
  },
}))

export const ROLE_LABELS: Record<UserRole, string> = {
  dispatch: "Dispatch",
  research: "Research",
  planning: "Planning",
  executive: "Executive",
  superadmin: "Superadmin",
}

export const ROLE_DESCRIPTIONS: Record<UserRole, string> = {
  dispatch: "Live operations & interventions",
  research: "Models, training & accuracy",
  planning: "Routes, schedules & demand",
  executive: "KPIs, ROI & financials",
  superadmin: "Full system access",
}
