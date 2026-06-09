import type { ReactNode } from "react";

interface TabsProps { children: ReactNode; className?: string }
interface TabProps { label: string; active?: boolean; onClick?: () => void }

export function Tabs({ children, className = "" }: TabsProps) {
  return <div className={`flex gap-1 ${className}`}>{children}</div>;
}
export function Tab({ label, active, onClick }: TabProps) {
  return <button onClick={onClick} className={`px-4 py-2 text-sm rounded-full font-semibold transition-all ${active ? "bg-michi-dark text-white shadow-sm" : "text-michi-body hover:bg-michi-warm hover:text-michi-dark"}`}>{label}</button>;
}