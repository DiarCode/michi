import type { ReactNode } from "react";

const ROLES = ["Dispatch Manager", "City Planner", "Executive"] as const;
type Role = typeof ROLES[number];

interface Props { children: (role: Role) => ReactNode; fallback?: ReactNode }

export default function RoleGuard({ children, fallback = null }: Props) {
  const role = (localStorage.getItem("michi_role") || "Dispatch Manager") as Role;
  return <>{children(role) ?? fallback}</>;
}

export { ROLES, type Role };
