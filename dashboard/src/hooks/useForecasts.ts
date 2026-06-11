import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"

export function useForecast(stationId: string) {
  return useQuery({
    queryKey: ["forecast", stationId],
    queryFn: () =>
      api.get(`/stations/${stationId}/forecast`).then((r) => r.data),
    enabled: !!stationId,
    refetchInterval: 300000,
  })
}
