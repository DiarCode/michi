import { useQuery } from "@tanstack/react-query"
import { fetchAlerts } from "@/lib/api"

export function useAlerts(severity?: string) {
  return useQuery({
    queryKey: ["alerts", severity],
    queryFn: () => fetchAlerts(),
  })
}