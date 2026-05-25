import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchAlerts, api } from "@/lib/api";

export function useAlerts(severity?: string) {
  return useQuery({
    queryKey: ["alerts", severity],
    queryFn: () => fetchAlerts(),
  });
}

export function useAckAlert() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.post(`/alerts/${id}/ack`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }),
  });
}
