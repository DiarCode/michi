import type { ReactNode } from "react";

interface Props { children: ReactNode; variant?: "default" | "success" | "warning" | "danger" | "outline"; className?: string }

const colors = {
  default: "bg-michi-warm text-michi-body border border-michi-border",
  success: "bg-michi-lime/20 text-michi-dark",
  warning: "bg-amber-100 text-amber-800",
  danger: "bg-red-50 text-michi-red",
  outline: "border border-michi-border text-michi-body",
};

export function Badge({ children, variant = "default", className = "" }: Props) {
  return <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${colors[variant]} ${className}`}>{children}</span>;
}