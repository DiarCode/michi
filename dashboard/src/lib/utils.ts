export function cn(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}

export function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toFixed(0);
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function severityColor(s: string): string {
  switch (s) {
    case "high": return "text-red-600 bg-red-50";
    case "medium": return "text-amber-600 bg-amber-50";
    case "low": return "text-blue-600 bg-blue-50";
    default: return "text-gray-600 bg-gray-50";
  }
}
