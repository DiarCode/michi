import { useState } from "react";
import { api } from "../lib/api";

export default function ScenarioPlanner() {
  const [result, setResult] = useState<any>(null);
  const runScenario = async () => {
    const { data } = await api.post("/scenarios/run", {
      name: "Frequency Increase",
      modifications: [{ type: "frequency", target: "Route_12", params: { headway: 5 } }],
    });
    setResult(data);
  };
  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Scenario Planner</h2>
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-bold mb-2">Configure Scenario</h3>
          <button onClick={runScenario} className="bg-blue-600 text-white px-4 py-2 rounded">Run Scenario</button>
        </div>
        {result && (
          <div className="bg-white p-4 rounded shadow">
            <h3 className="font-bold mb-2">Results</h3>
            <p>Ridership change: {result.changes.ridership}%</p>
            <p>Wait time change: {result.changes.avg_wait}%</p>
          </div>
        )}
      </div>
    </div>
  );
}
