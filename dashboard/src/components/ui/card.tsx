import type { ReactNode } from "react";

interface CardProps { children: ReactNode; className?: string }
export function Card({ children, className = "" }: CardProps) {
  return <div className={`rounded-2xl border border-michi-border bg-white shadow-card ${className}`}>{children}</div>;
}
export function CardHeader({ children, className = "" }: CardProps) {
  return <div className={`p-6 pb-2 ${className}`}>{children}</div>;
}
export function CardTitle({ children, className = "" }: CardProps) {
  return <h3 className={`font-bold text-lg text-michi-dark ${className}`}>{children}</h3>;
}
export function CardContent({ children, className = "" }: CardProps) {
  return <div className={`p-6 pt-2 ${className}`}>{children}</div>;
}