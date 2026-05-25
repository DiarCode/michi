import { useQuery } from "@tanstack/react-query";
import { fetchStations } from "@/lib/api";

export function useStations() {
  return useQuery({
    queryKey: ["stations"],
    queryFn: fetchStations,
    refetchInterval: 60000,
  });
}
