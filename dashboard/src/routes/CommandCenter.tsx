import KPIGrid from "@/components/dashboard/KPIGrid";
import CongestionHeatmap from "@/components/dashboard/CongestionHeatmap";
import AlertTicker from "@/components/dashboard/AlertTicker";

export default function CommandCenter() {
  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Command Center</h2>
      <KPIGrid />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CongestionHeatmap />
        <AlertTicker />
      </div>
    </div>
  );
}
