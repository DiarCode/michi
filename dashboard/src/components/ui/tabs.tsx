import type { ReactNode } from "react";

interface TabsProps { children: ReactNode; className?: string }
interface TabProps { label: string; active?: boolean; onClick?: () => void }

export function Tabs({ children, className = "" }: TabsProps) {
  return <div className={`flex border-b ${className}`}>{children}</div>;
}
export function Tab({ label, active, onClick }: TabProps) {
  return <button onClick={onClick} className={`px-4 py-2 text-sm font-medium border-b-2 transition ${active ? "border-blue-600 text-blue-600" : "border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"}`}>{label}</button>;
}
