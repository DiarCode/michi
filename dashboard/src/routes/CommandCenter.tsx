import { useQuery } from "@tanstack/react-query";
import { fetchKPIs } from "../lib/api";

export default function CommandCenter() {
  const { data: kpis } = useQuery({ queryKey: ["kpis"], queryFn: fetchKPIs });
  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Command Center</h2>
      <div className="grid grid-cols-4 gap-4">
        {[{label:"Stations",value:kpis?.total_stations},{label:"Active Routes",value:kpis?.active_routes},{label:"Avg Ridership",value:kpis?.avg_ridership},{label:"Alerts Today",value:kpis?.alerts_today}].map((kpi)=>(
          <div key={kpi.label} className="bg-white p-4 rounded shadow">
            <div className="text-sm text-gray-500">{kpi.label}</div>
            <div className="text-3xl font-bold">{kpi.value ?? "—"}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
