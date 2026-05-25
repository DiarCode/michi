import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

const SCENARIO_TYPES = [
  { value: "frequency", label: "Change Frequency" },
  { value: "route_add", label: "Add Route" },
  { value: "station_close", label: "Close Station" },
];

export default function ScenarioPlanner() {
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [scenarioType, setScenarioType] = useState("frequency");
  const [loading, setLoading] = useState(false);

  const runScenario = async () => {
    setLoading(true);
    try {
      const { data } = await api.post("/scenarios/run", {
        name: scenarioType,
        modifications: [{ type: scenarioType, target: "R1", params: { headway: 5 } }],
      });
      setResult(data);
    } finally { setLoading(false); }
  };

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Scenario Planner</h2>
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
              <Button onClick={runScenario} disabled={loading}>{loading ? "Running..." : "Run Scenario"}</Button>
            </div>
          </CardContent>
        </Card>
        {result && (
          <Card>
            <CardHeader><CardTitle>Results</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <div><p className="text-sm text-gray-500">Base Ridership</p><p className="text-xl font-bold">{(result.base_metrics as Record<string, number>).ridership?.toLocaleString()}</p></div>
                  <div><p className="text-sm text-gray-500">Scenario Ridership</p><p className="text-xl font-bold">{(result.scenario_metrics as Record<string, number>).ridership?.toLocaleString()}</p></div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div><p className="text-sm text-gray-500">Ridership Change</p><p className={`text-xl font-bold ${(result.changes as Record<string, number>).ridership >= 0 ? "text-green-600" : "text-red-600"}`}>{(result.changes as Record<string, number>).ridership}%</p></div>
                  <div><p className="text-sm text-gray-500">Wait Time Change</p><p className={`text-xl font-bold ${(result.changes as Record<string, number>).avg_wait >= 0 ? "text-red-600" : "text-green-600"}`}>{(result.changes as Record<string, number>).avg_wait}%</p></div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
