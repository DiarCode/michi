import type { ReactNode } from "react";
import type { UserRole } from "../../types";

const ROLES: UserRole[] = ["dispatch", "research", "planning", "executive", "superadmin"];

interface Props { children: (role: UserRole) => ReactNode; fallback?: ReactNode }

export default function RoleGuard({ children, fallback = null }: Props) {
  const role = (localStorage.getItem("michi-role") || "dispatch") as UserRole;
  return <>{children(role) ?? fallback}</>;
}

export { ROLES };