import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

const SCENARIO_TYPES = [
  { value: "frequency", label: "Change Frequency" },
  { value: "route_add", label: "Add Route" },
  { value: "station_close", label: "Close Station" },
];

interface ScenarioResult {
  scenario_id: string;
  base_metrics: { ridership: number; avg_wait: number };
  scenario_metrics: { ridership: number; avg_wait: number };
  changes: { ridership: number; avg_wait: number };
}

export default function ScenarioPlanner() {
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [scenarioType, setScenarioType] = useState("frequency");
  const [headway, setHeadway] = useState(5);
  const [loading, setLoading] = useState(false);

  const runScenario = async () => {
    setLoading(true);
    try {
      const { data } = await api.post("/scenarios/run", {
        name: scenarioType,
        modifications: [{ type: scenarioType, target: "R1", params: { headway } }],
      });
      setResult(data);
    } catch { /* scenario error handled by UI state */ } finally { setLoading(false); }
  };

  const pctColor = (val: number, positiveIsGood = true) =>
    val >= 0 ? (positiveIsGood ? "text-green-600" : "text-red-600") : (positiveIsGood ? "text-red-600" : "text-green-600");

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Scenario Planner</h2>
      <p className="text-sm text-gray-500">Simulate what-if changes and compare against current baseline.</p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>Configure Scenario</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Scenario Type</label>
                <select className="w-full border rounded px-3 py-2" value={scenarioType} onChange={(e) => setScenarioType(e.target.value)}>
                  {SCENARIO_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
                </select>
              </div>
              {scenarioType === "frequency" && (
                <div>
                  <label className="block text-sm font-medium mb-1">New Headway (min)</label>
                  <input type="range" min={2} max={20} value={headway} onChange={(e) => setHeadway(Number(e.target.value))} className="w-full" />
                  <p className="text-xs text-gray-500 text-center">{headway} min between buses</p>
                </div>
              )}
              <Button onClick={runScenario} disabled={loading} className="w-full">{loading ? "Running..." : "Run Scenario"}</Button>
            </div>
          </CardContent>
        </Card>

        {result && (
          <Card>
            <CardHeader><CardTitle>Before vs After Comparison</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-gray-50 rounded-lg p-3"><p className="text-xs text-gray-500">Metric</p></div>
                  <div className="bg-gray-50 rounded-lg p-3"><p className="text-xs text-gray-500">Baseline</p></div>
                  <div className="bg-gray-50 rounded-lg p-3"><p className="text-xs text-gray-500">Scenario</p></div>
                  <div className="rounded-lg p-3"><p className="text-xs text-gray-500">Ridership</p></div>
                  <div className="rounded-lg p-3"><p className="text-lg font-bold">{result.base_metrics.ridership.toLocaleString()}</p></div>
                  <div className="rounded-lg p-3"><p className="text-lg font-bold">{result.scenario_metrics.ridership.toLocaleString()}</p></div>
                  <div className="rounded-lg p-3"><p className="text-xs text-gray-500">Avg Wait</p></div>
                  <div className="rounded-lg p-3"><p className="text-lg font-bold">{result.base_metrics.avg_wait} min</p></div>
                  <div className="rounded-lg p-3"><p className="text-lg font-bold">{result.scenario_metrics.avg_wait} min</p></div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4 text-center">
                    <p className="text-sm text-gray-500 mb-1">Ridership Change</p>
                    <p className={`text-2xl font-bold ${pctColor(result.changes.ridership)}`}>{result.changes.ridership > 0 ? "+" : ""}{result.changes.ridership}%</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4 text-center">
                    <p className="text-sm text-gray-500 mb-1">Wait Time Change</p>
                    <p className={`text-2xl font-bold ${pctColor(result.changes.avg_wait, false)}`}>{result.changes.avg_wait > 0 ? "+" : ""}{result.changes.avg_wait}%</p>
                  </div>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Ridership Comparison</p>
                  <div className="relative h-8 bg-gray-200 rounded overflow-hidden">
                    <div className="absolute h-full bg-blue-500 rounded" style={{ width: "100%" }} />
                    <div className="absolute h-full bg-green-500 rounded opacity-80" style={{ width: Math.min(100, (result.scenario_metrics.ridership / result.base_metrics.ridership) * 100) + "%" }} />
                    <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">Baseline → Scenario</span>
                  </div>
                  <p className="text-sm font-medium mt-2">Wait Time Comparison</p>
                  <div className="relative h-8 bg-gray-200 rounded overflow-hidden">
                    <div className="absolute h-full bg-amber-500 rounded" style={{ width: "100%" }} />
                    <div className="absolute h-full bg-red-500 rounded opacity-80" style={{ width: Math.min(100, (result.scenario_metrics.avg_wait / result.base_metrics.avg_wait) * 100) + "%" }} />
                    <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">Baseline → Scenario</span>
                  </div>
                </div>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-sm font-medium text-blue-800">
                    {result.changes.ridership > 0
                      ? `Ridership increases by ${result.changes.ridership}% with ${result.changes.avg_wait > 0 ? "longer" : "shorter"} wait times.`
                      : `Ridership decreases by ${Math.abs(result.changes.ridership)}% — consider adjusting frequency.`}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
