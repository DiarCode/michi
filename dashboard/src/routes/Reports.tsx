import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { showToast } from "@/lib/toast";

interface OperationsReport {
  date: string;
  kpis: Record<string, number | string>;
  district_summary: Record<string, { stations: number; total_ridership: number }>;
  peak_hours: string[];
  over_capacity_stations: { id: string; name: string; ridership_24h: number }[];
  total_stations: number;
}

export default function Reports() {
  const [report, setReport] = useState<OperationsReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateReport = async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await api.get("/dashboard/operations");
      setReport(data);
    } catch (e: any) {
      setError(e.response?.data?.detail || "Failed to generate report");
    } finally {
      setLoading(false);
    }
  };

  const exportCsv = async () => {
    try {
      const { data } = await api.get("/dashboard/operations", {
        params: { format: "csv" },
        responseType: "blob",
      });
      const url = URL.createObjectURL(new Blob([data], { type: "text/csv" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `operations_${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: any) { showToast.error(`CSV export failed: ${err.message}`); }
  };

  const loadColor = (pct: number) =>
    pct >= 95 ? "text-red-600" : pct >= 85 ? "text-amber-600" : "text-green-600";

  const loadBg = (pct: number) =>
    pct >= 95 ? "bg-red-50" : pct >= 85 ? "bg-amber-50" : "bg-green-50";

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Operations Report</h2>
        <div className="flex gap-2">
          <Button onClick={generateReport} disabled={loading}>
            {loading ? "Generating..." : "Generate Report"}
          </Button>
          {report && (
            <Button variant="outline" onClick={exportCsv}>
              Export CSV
            </Button>
          )}
        </div>
      </div>

      {error && <p className="text-red-600 text-sm">{error}</p>}

      {report && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(report.kpis).map(([k, v]) => (
              <Card key={k}>
                <CardContent className="p-4 text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-wide">{k.replace(/_/g, " ")}</p>
                  <p className="text-2xl font-bold mt-1">
                    {typeof v === "number" ? (v >= 100 ? v.toLocaleString() : v.toFixed(1)) : v}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader><CardTitle>District Summary</CardTitle></CardHeader>
            <CardContent>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">District</th>
                    <th className="text-right py-2">Stations</th>
                    <th className="text-right py-2">Total Ridership</th>
                    <th className="text-right py-2">Avg per Station</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(report.district_summary).map(([d, vals]) => (
                    <tr key={d} className="border-b hover:bg-gray-50">
                      <td className="py-2 font-medium">{d}</td>
                      <td className="text-right py-2">{vals.stations}</td>
                      <td className="text-right py-2 font-mono">{vals.total_ridership.toLocaleString()}</td>
                      <td className="text-right py-2 font-mono">
                        {vals.stations > 0 ? Math.round(vals.total_ridership / vals.stations).toLocaleString() : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>

          {report.over_capacity_stations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-red-700">Over-Capacity Stations</CardTitle>
              </CardHeader>
              <CardContent>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Station</th>
                      <th className="text-right py-2">24h Ridership</th>
                      <th className="text-right py-2">Load Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.over_capacity_stations.map((s) => {
                      const pct = Math.round((s.ridership_24h / 4000) * 100);
                      return (
                        <tr key={s.id} className={"border-b " + loadBg(pct)}>
                          <td className="py-2 font-medium">{s.name || s.id}</td>
                          <td className="text-right py-2 font-mono">{s.ridership_24h.toLocaleString()}</td>
                          <td className={"text-right py-2 font-bold " + loadColor(pct)}>{pct}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader><CardTitle>Peak Hours</CardTitle></CardHeader>
            <CardContent>
              <div className="flex gap-3 flex-wrap">
                {report.peak_hours.map((h) => (
                  <span key={h} className="bg-blue-100 text-blue-800 px-3 py-1.5 rounded-full text-sm font-medium">{h}</span>
                ))}
              </div>
            </CardContent>
          </Card>

          <p className="text-xs text-gray-400">Report generated for {report.date} — {report.total_stations} stations</p>
        </>
      )}
    </div>
  );
}