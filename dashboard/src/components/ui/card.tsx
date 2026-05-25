import type { ReactNode } from "react";

interface CardProps { children: ReactNode; className?: string }
export function Card({ children, className = "" }: CardProps) {
  return <div className={`rounded-xl border bg-white shadow-sm ${className}`}>{children}</div>;
}
export function CardHeader({ children, className = "" }: CardProps) {
  return <div className={`p-4 pb-2 ${className}`}>{children}</div>;
}
export function CardTitle({ children, className = "" }: CardProps) {
  return <h3 className={`font-semibold text-lg ${className}`}>{children}</h3>;
}
export function CardContent({ children, className = "" }: CardProps) {
  return <div className={`p-4 pt-0 ${className}`}>{children}</div>;
}
