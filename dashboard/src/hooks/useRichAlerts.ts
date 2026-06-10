import { useQuery } from "@tanstack/react-query";
import { fetchRichAlerts } from "@/lib/api";

/** Fetch rich alerts with 15-second auto-refresh */
export function useRichAlerts(enabled = true) {
  const query = useQuery({
    queryKey: ["alerts", "rich"],
    queryFn: fetchRichAlerts,
    refetchInterval: 15_000,
    enabled,
    select: (data) => data.alerts ?? [],
  });

  const alerts = query.data ?? [];

  return {
    alerts,
    criticalAlerts: alerts.filter((a) => a.severity === "critical"),
    unacknowledgedAlerts: alerts.filter((a) => !a.acknowledged),
    alertsByFamily: groupBy(alerts, (a) => a.family ?? "unknown"),
    isLoading: query.isLoading,
  };
}

function groupBy<T>(arr: T[], key: (item: T) => string): Record<string, T[]> {
  const map: Record<string, T[]> = {};
  for (const item of arr) {
    const k = key(item);
    if (!map[k]) map[k] = [];
    map[k].push(item);
  }
  return map;
}