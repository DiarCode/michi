import { useState, useEffect, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { showToast } from "@/lib/toast";
import { useSimulationStore } from "@/stores/simulationStore";
import { useConnectionStore } from "@/stores/connectionStore";
import SimulationMetrics from "@/components/dashboard/SimulationMetrics";
import { Play, Square, Activity, Clock, Users, BarChart3, TrendingUp, AlertTriangle } from "lucide-react";

interface StationSimData {
  station_id: string;
  name: string;
  actual: number;
  predicted: number;
  confidence: number;
  confidence_upper: number;
  confidence_lower: number;
  error_pct: number;
}

export default function SimulationPage() {
  const [startingSim, setStartingSim] = useState(false);
  const [stoppingSim, setStoppingSim] = useState(false);
  const [stationData, setStationData] = useState<StationSimData[]>([]);
  const [simState, setSimState] = useState<{
    running: boolean;
    tick: number;
    drift_status: string;
    current_time: string | null;
    station_count: number | null;
    metrics: { mae: number | null; mape: number | null; accuracy: number | null };
  } | null>(null);

  const { running, tick, subscribe, startSimulation: storeStart, stopSimulation: storeStop } = useSimulationStore();
  const connected = useConnectionStore((s) => s.connected);

  useEffect(() => {
    const unsub = subscribe();
    return unsub;
  }, [subscribe]);

  const fetchSimState = useCallback(async () => {
    try {
      const { data } = await api.get("/simulation/state");
      setSimState(data);
      if (data.running && !running) {
        storeStart();
      }
    } catch {
      // API not available
    }
  }, [running, storeStart]);

  useEffect(() => {
    fetchSimState();
    const interval = setInterval(fetchSimState, 5000);
    return () => clearInterval(interval);
  }, [fetchSimState]);

  const fetchStationData = useCallback(async () => {
    if (!running && !simState?.running) return;
    try {
      const { data } = await api.get("/simulation/station-data");
      if (data?.stations) {
        setStationData(
          Object.entries(data.stations).map(([id, s]: [string, any]) => ({
            station_id: id,
            name: s.name ?? id,
            actual: s.actual ?? 0,
            predicted: s.predicted ?? 0,
            confidence: s.confidence ?? 0,
            confidence_upper: s.confidence_upper ?? 0,
            confidence_lower: s.confidence_lower ?? 0,
            error_pct: s.actual > 0 ? Math.abs(s.predicted - s.actual) / s.actual * 100 : 0,
          }))
        );
      }
    } catch {
      // Endpoint may not be available yet
    }
  }, [running, simState?.running]);

  useEffect(() => {
    if (!running && !simState?.running) return;
    fetchStationData();
    const interval = setInterval(fetchStationData, 5000);
    return () => clearInterval(interval);
  }, [fetchStationData]);

  const handleStart = async () => {
    setStartingSim(true);
    try {
      const { data } = await api.post("/simulation/start");
      if (data.status === "started" || data.status === "already_running") {
        storeStart();
        showToast.success("Simulation engine started");
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

  const driftColor = (status: string) => {
    if (status === "critical") return "bg-red-500 text-white";
    if (status === "warning") return "bg-amber-500 text-white";
    return "bg-green-500 text-white";
  };

  const errorColor = (pct: number) => {
    if (pct > 15) return "text-red-600 dark:text-red-400";
    if (pct > 8) return "text-amber-600 dark:text-amber-400";
    return "text-green-600 dark:text-green-400";
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold dark:text-white">Simulation</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">Real-time DTS-GSSF model validation and passenger flow simulation across all stations.</p>
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

      {/* Engine controls + status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Engine Control</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Button onClick={handleStart} disabled={running || startingSim} className="flex items-center gap-2">
              <Play className="h-4 w-4" />
              {startingSim ? "Starting..." : "Start Engine"}
            </Button>
            <Button onClick={handleStop} disabled={!running || stoppingSim} variant="destructive" className="flex items-center gap-2">
              <Square className="h-4 w-4" />
              {stoppingSim ? "Stopping..." : "Stop Engine"}
            </Button>
            {simState && (
              <div className="ml-auto flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                <span className="flex items-center gap-1"><Clock className="h-3 w-3" /> {simState.current_time ?? "—"}</span>
                <span className="flex items-center gap-1"><Users className="h-3 w-3" /> {simState.station_count ?? 0} stations</span>
                <Badge className={driftColor(simState.drift_status)}>
                  Drift: {simState.drift_status.toUpperCase()}
                </Badge>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Real-time validation metrics */}
      <SimulationMetrics />

      {/* Station-level simulation data */}
      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle className="text-sm">Station-Level Simulation Data</CardTitle>
          {stationData.length > 0 && (
            <span className="text-xs text-gray-400">Updates every 5s · {stationData.length} stations</span>
          )}
        </CardHeader>
        <CardContent>
          {!running && !simState?.running ? (
            <div className="text-center py-12">
              <Activity className="h-12 w-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
              <p className="text-gray-500 dark:text-gray-400 text-sm">Start the simulation engine to view real-time station data.</p>
              <p className="text-gray-400 text-xs mt-1">Data refreshes every 5 seconds with predicted vs actual passenger counts.</p>
            </div>
          ) : stationData.length === 0 ? (
            <div className="text-center py-8">
              <div className="animate-pulse flex flex-col items-center gap-2">
                <Activity className="h-8 w-8 text-blue-400" />
                <p className="text-sm text-gray-400">Waiting for simulation data...</p>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b dark:border-gray-700 text-left">
                    <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400">Stop Name</th>
                    <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400 text-right">Actual Passengers</th>
                    <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400 text-right">Forecast</th>
                    <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400 text-right">Accuracy</th>
                    <th className="pb-2 pr-4 font-semibold text-gray-500 dark:text-gray-400 text-right">Forecast Range</th>
                    <th className="pb-2 font-semibold text-gray-500 dark:text-gray-400 text-right">Error Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {stationData.slice(0, 20).map((s) => (
                    <tr key={s.station_id} className="border-b dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                      <td className="py-2 pr-4 font-medium dark:text-white truncate max-w-[200px]">{s.name}</td>
                      <td className="py-2 pr-4 text-right font-mono dark:text-gray-300">{s.actual.toLocaleString()}</td>
                      <td className="py-2 pr-4 text-right font-mono dark:text-gray-300">{s.predicted.toLocaleString()}</td>
                      <td className="py-2 pr-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <div className="w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 rounded-full" style={{ width: `${s.confidence * 100}%` }} />
                          </div>
                          <span className="text-xs font-mono text-gray-500">{(s.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="py-2 pr-4 text-right font-mono text-xs text-gray-400">
                        {s.confidence_lower.toLocaleString()} – {s.confidence_upper.toLocaleString()}
                      </td>
                      <td className={`py-2 text-right font-mono font-medium ${errorColor(s.error_pct)}`}>
                        {s.error_pct.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {stationData.length > 20 && (
                <p className="text-xs text-gray-400 mt-2 text-center">Showing top 20 of {stationData.length} stations</p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Summary cards */}
      {stationData.length > 0 && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                <Users className="h-3 w-3" /> Total Actual Passengers
              </div>
              <p className="text-xl font-bold dark:text-white">
                {stationData.reduce((sum, s) => sum + s.actual, 0).toLocaleString()}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                <BarChart3 className="h-3 w-3" /> Total Forecasted
              </div>
              <p className="text-xl font-bold dark:text-white">
                {stationData.reduce((sum, s) => sum + s.predicted, 0).toLocaleString()}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                <TrendingUp className="h-3 w-3" /> Average Accuracy
              </div>
              <p className="text-xl font-bold text-blue-600 dark:text-blue-400">
                {(stationData.reduce((sum, s) => sum + s.confidence, 0) / stationData.length * 100).toFixed(1)}%
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                <AlertTriangle className="h-3 w-3" /> Stops with High Error
              </div>
              <p className={`text-xl font-bold ${stationData.filter(s => s.error_pct > 15).length > 0 ? "text-red-600 dark:text-red-400" : "text-green-600 dark:text-green-400"}`}>
                {stationData.filter(s => s.error_pct > 15).length} / {stationData.length}
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}