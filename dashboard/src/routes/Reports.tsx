import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { showToast } from "@/lib/toast";
import { FileText, Download } from "lucide-react";

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
    pct >= 95 ? "text-michi-red" : pct >= 85 ? "text-michi-amber" : "text-michi-lime-dark";

  const loadBg = (pct: number) =>
    pct >= 95 ? "bg-michi-red/5" : pct >= 85 ? "bg-michi-amber/5" : "bg-michi-lime/5";

  return (
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold text-michi-dark">Operations Report</h1>
          <p className="text-base text-michi-muted mt-1">Generate and export operational summaries</p>
        </div>
        <div className="flex gap-3">
          <Button onClick={generateReport} disabled={loading} variant="lime">
            <FileText size={16} className="mr-1.5" />
            {loading ? "Generating..." : "Generate Report"}
          </Button>
          {report && (
            <Button variant="outline" onClick={exportCsv}>
              <Download size={16} className="mr-1.5" />
              Export CSV
            </Button>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-michi-red/10 border border-michi-red/30 text-michi-red rounded-xl p-4 text-sm font-semibold">
          {error}
        </div>
      )}

      {report && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
            {Object.entries(report.kpis).map(([k, v]) => (
              <Card key={k}>
                <CardContent className="p-5 text-center">
                  <p className="text-sm text-michi-muted font-medium uppercase tracking-wide">{k.replace(/_/g, " ")}</p>
                  <p className="text-3xl font-extrabold text-michi-dark mt-2">
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
                  <tr className="border-b border-michi-border">
                    <th className="text-left py-2.5 font-semibold text-michi-muted">District</th>
                    <th className="text-right py-2.5 font-semibold text-michi-muted">Stations</th>
                    <th className="text-right py-2.5 font-semibold text-michi-muted">Total Ridership</th>
                    <th className="text-right py-2.5 font-semibold text-michi-muted">Avg per Station</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(report.district_summary).map(([d, vals]) => (
                    <tr key={d} className="border-b border-michi-border/50 hover:bg-michi-warm transition-colors">
                      <td className="py-2.5 font-semibold text-michi-dark">{d}</td>
                      <td className="text-right py-2.5 text-michi-body">{vals.stations}</td>
                      <td className="text-right py-2.5 font-mono text-michi-body">{vals.total_ridership.toLocaleString()}</td>
                      <td className="text-right py-2.5 font-mono text-michi-body">
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
                <CardTitle className="text-michi-red">Over-Capacity Stations</CardTitle>
              </CardHeader>
              <CardContent>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-michi-border">
                      <th className="text-left py-2.5 font-semibold text-michi-muted">Station</th>
                      <th className="text-right py-2.5 font-semibold text-michi-muted">24h Ridership</th>
                      <th className="text-right py-2.5 font-semibold text-michi-muted">Load Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.over_capacity_stations.map((s) => {
                      const pct = Math.round((s.ridership_24h / 4000) * 100);
                      return (
                        <tr key={s.id} className={"border-b border-michi-border/50 " + loadBg(pct)}>
                          <td className="py-2.5 font-semibold text-michi-dark">{s.name || s.id}</td>
                          <td className="text-right py-2.5 font-mono text-michi-body">{s.ridership_24h.toLocaleString()}</td>
                          <td className={"text-right py-2.5 font-mono font-bold " + loadColor(pct)}>{pct}%</td>
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
                  <span key={h} className="bg-michi-lime/15 text-michi-lime-dark px-4 py-2 rounded-full text-sm font-semibold">{h}</span>
                ))}
              </div>
            </CardContent>
          </Card>

          <p className="text-sm text-michi-muted font-medium">Report generated for {report.date} — {report.total_stations} stations</p>
        </>
      )}
    </div>
  );
}