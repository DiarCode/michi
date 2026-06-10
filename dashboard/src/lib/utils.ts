import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(n: number, opts?: Intl.NumberFormatOptions) {
  return new Intl.NumberFormat("en-US", opts).format(n)
}

export function formatPercent(n: number, fractionDigits = 1) {
  return `${(n * 100).toFixed(fractionDigits)}%`
}

export function formatTime(date: Date | string) {
  const d = typeof date === "string" ? new Date(date) : date
  return d.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
}
