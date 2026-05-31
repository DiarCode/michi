import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { showToast } from "@/lib/toast";
import { useSimulationStore } from "@/stores/simulationStore";
import { useConnectionStore } from "@/stores/connectionStore";
import SimulationMetrics from "@/components/dashboard/SimulationMetrics";
import { Play, Square, Activity } from "lucide-react";

const SIMULATION_TYPES = [
  { value: "frequency", label: "Frequency Change" },
  { value: "route_add", label: "Add Route" },
  { value: "station_close", label: "Close Station" },
  { value: "demand_surge", label: "Demand Surge" },
];

interface SimResult {
  scenario_id: string;
  base_metrics: { ridership: number; avg_wait: number };
  scenario_metrics: { ridership: number; avg_wait: number };
  changes: { ridership: number; avg_wait: number };
}

export default function SimulationPage() {
  const [simType, setSimType] = useState("frequency");
  const [headway, setHeadway] = useState(5);
  const [result, setResult] = useState<SimResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [startingSim, setStartingSim] = useState(false);
  const [stoppingSim, setStoppingSim] = useState(false);

  const { running, tick, subscribe, startSimulation: storeStart, stopSimulation: storeStop } = useSimulationStore();
  const connected = useConnectionStore((s) => s.connected);

  // Subscribe to simulation WS events
  useEffect(() => {
    const unsub = subscribe();
    return unsub;
  }, [subscribe]);

  // Fetch initial simulation state on mount
  useEffect(() => {
    api.get("/simulation/state").then((res) => {
      if (res.data?.running) {
        storeStart();
      }
    }).catch(() => { /* ignore if API not available */ });
  }, [storeStart]);

  const handleStart = async () => {
    setStartingSim(true);
    try {
      const { data } = await api.post("/simulation/start");
      if (data.status === "started" || data.status === "already_running") {
        storeStart();
        showToast.success("Simulation started");
      }
    } catch (err: any) {
      showToast.error(`Failed to start simulation: ${err.message}`);
    } finally {
      setStartingSim(false);
    }
  };

  const handleStop = async () => {
    setStoppingSim(true);
    try {
      await api.post("/simulation/stop");
      storeStop();
      showToast.info("Simulation stopped");
    } catch (err: any) {
      showToast.error(`Failed to stop simulation: ${err.message}`);
    } finally {
      setStoppingSim(false);
    }
  };

  const runSim = async () => {
    setLoading(true);
    try {
      const { data } = await api.post("/scenarios/run", {
        name: simType,
        modifications: [{ type: simType, target: "R12", params: { headway } }],
      });
      setResult(data);
    } catch (err: any) { showToast.error(`Simulation failed: ${err.message}`); } finally { setLoading(false); }
  };

  const pctColor = (val: number, positiveIsGood = true) =>
    val >= 0 ? (positiveIsGood ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400") : (positiveIsGood ? "text-red-600 dark:text-red-400" : "text-green-600 dark:text-green-400");

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold dark:text-white">Simulation</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">Run transit simulations and monitor real-time model validation.</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="default" className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
            {connected ? "Live" : "Offline"}
          </Badge>
          {running && (
            <Badge variant="default" className="flex items-center gap-1.5 text-blue-600 border-blue-300">
              <Activity className="h-3 w-3" />
              Tick #{tick}
            </Badge>
          )}
        </div>
      </div>

      {/* Simulation engine controls */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Engine Control</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3">
            <Button
              onClick={handleStart}
              disabled={running || startingSim}
              className="flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              {startingSim ? "Starting..." : "Start Engine"}
            </Button>
            <Button
              onClick={handleStop}
              disabled={!running || stoppingSim}
              variant="destructive"
              className="flex items-center gap-2"
            >
              <Square className="h-4 w-4" />
              {stoppingSim ? "Stopping..." : "Stop Engine"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Real-time validation metrics */}
      <SimulationMetrics />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Scenario planner */}
        <Card>
          <CardHeader><CardTitle>Configure Scenario</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Simulation Type</label>
              <select className="w-full border rounded px-3 py-2 dark:bg-gray-800 dark:text-gray-100" value={simType} onChange={(e) => setSimType(e.target.value)}>
                {SIMULATION_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
            {simType === "frequency" && (
              <div>
                <label className="block text-sm font-medium mb-1">New Headway (min)</label>
                <input type="range" min={2} max={20} value={headway} onChange={(e) => setHeadway(Number(e.target.value))} className="w-full" />
                <p className="text-xs text-gray-500 dark:text-gray-400 text-center">{headway} min between buses</p>
              </div>
            )}
            {simType === "demand_surge" && (
              <div>
                <label className="block text-sm font-medium mb-1">Surge Factor</label>
                <input type="range" min={110} max={200} value={headway * 10} onChange={(e) => setHeadway(Math.round(Number(e.target.value) / 10))} className="w-full" />
                <p className="text-xs text-gray-500 dark:text-gray-400 text-center">{Math.round(headway * 10)}% of normal demand</p>
              </div>
            )}
            <Button onClick={runSim} disabled={loading} className="w-full">{loading ? "Running..." : "Run Scenario"}</Button>
          </CardContent>
        </Card>

        {result && (
          <Card>
            <CardHeader><CardTitle>Impact Projection</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-3 text-center">
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3"><p className="text-xs text-gray-500 dark:text-gray-400">Metric</p></div>
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3"><p className="text-xs text-gray-500 dark:text-gray-400">Baseline</p></div>
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3"><p className="text-xs text-gray-500 dark:text-gray-400">Simulated</p></div>
                <div className="rounded-lg p-3"><p className="text-xs text-gray-500 dark:text-gray-400">Ridership</p></div>
                <div className="rounded-lg p-3"><p className="text-lg font-bold dark:text-white">{result.base_metrics.ridership.toLocaleString()}</p></div>
                <div className="rounded-lg p-3"><p className="text-lg font-bold dark:text-white">{result.scenario_metrics.ridership.toLocaleString()}</p></div>
                <div className="rounded-lg p-3"><p className="text-xs text-gray-500 dark:text-gray-400">Avg Wait</p></div>
                <div className="rounded-lg p-3"><p className="text-lg font-bold dark:text-white">{result.base_metrics.avg_wait} min</p></div>
                <div className="rounded-lg p-3"><p className="text-lg font-bold dark:text-white">{result.scenario_metrics.avg_wait} min</p></div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 text-center">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Ridership Change</p>
                  <p className={`text-2xl font-bold ${pctColor(result.changes.ridership)}`}>{result.changes.ridership > 0 ? "+" : ""}{result.changes.ridership}%</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 text-center">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Wait Time Change</p>
                  <p className={`text-2xl font-bold ${pctColor(result.changes.avg_wait, false)}`}>{result.changes.avg_wait > 0 ? "+" : ""}{result.changes.avg_wait}%</p>
                </div>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
                <p className="text-sm font-medium text-blue-800 dark:text-blue-300">
                  {result.changes.ridership > 0
                    ? `Ridership increases by ${result.changes.ridership}% with ${result.changes.avg_wait > 0 ? "longer" : "shorter"} wait times.`
                    : `Ridership decreases by ${Math.abs(result.changes.ridership)}% — consider adjusting parameters.`}
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Simulation type cards */}
      <Card>
        <CardHeader><CardTitle>Simulation Types</CardTitle></CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {SIMULATION_TYPES.map((t) => (
              <button key={t.value} onClick={() => setSimType(t.value)}
                className={`p-4 rounded-lg border-2 text-left transition-colors ${simType === t.value ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20" : "border-gray-200 dark:border-gray-700 hover:border-gray-300"}`}>
                <p className="font-medium text-sm dark:text-white">{t.label}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {t.value === "frequency" && "Adjust bus frequency and headway"}
                  {t.value === "route_add" && "Add a new route to the network"}
                  {t.value === "station_close" && "Simulate station closure impact"}
                  {t.value === "demand_surge" && "Model demand surge scenarios"}
                </p>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}