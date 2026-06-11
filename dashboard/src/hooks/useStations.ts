import { useQuery } from "@tanstack/react-query"
import { fetchStations } from "@/lib/api"

export function useStations(hour?: number) {
  return useQuery({
    queryKey: hour !== undefined ? ["stations", hour] : ["stations"],
    queryFn: () => fetchStations(hour),
    refetchInterval: 60000,
  })
}
