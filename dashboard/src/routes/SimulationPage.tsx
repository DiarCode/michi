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

  useEffect(() => { const unsub = subscribe(); return unsub; }, [subscribe]);

  const fetchSimState = useCallback(async () => {
    try {
      const { data } = await api.get("/simulation/state");
      setSimState(data);
      if (data.running && !running) storeStart();
    } catch { /* API not available */ }
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
            station_id: id, name: s.name ?? id,
            actual: s.actual ?? 0, predicted: s.predicted ?? 0,
            confidence: s.confidence ?? 0,
            confidence_upper: s.confidence_upper ?? 0,
            confidence_lower: s.confidence_lower ?? 0,
            error_pct: s.actual > 0 ? Math.abs(s.predicted - s.actual) / s.actual * 100 : 0,
          }))
        );
      }
    } catch { /* Endpoint may not be available yet */ }
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
    } catch (err: any) { showToast.error(`Failed to start simulation: ${err.message}`); }
    finally { setStartingSim(false); }
  };

  const handleStop = async () => {
    setStoppingSim(true);
    try {
      await api.post("/simulation/stop");
      storeStop();
      showToast.info("Simulation stopped");
    } catch (err: any) { showToast.error(`Failed to stop simulation: ${err.message}`); }
    finally { setStoppingSim(false); }
  };

  const driftBadge = (status: string) => {
    if (status === "critical") return "bg-michi-red text-white";
    if (status === "warning") return "bg-michi-amber text-white";
    return "bg-michi-lime text-michi-dark";
  };

  const errorColor = (pct: number) => {
    if (pct > 15) return "text-michi-red";
    if (pct > 8) return "text-michi-amber";
    return "text-michi-lime-dark";
  };

  return (
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold text-michi-dark">Simulation</h1>
          <p className="text-base text-michi-muted mt-1">Real-time DTS-GSSF model validation and passenger flow simulation across all stations</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="default" className="flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full ${connected ? "bg-michi-lime" : "bg-michi-red"}`} />
            {connected ? "Live" : "Offline"}
          </Badge>
          {running && (
            <Badge variant="success" className="flex items-center gap-2">
              <Activity size={14} />
              Tick #{tick}
            </Badge>
          )}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Engine Control</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Button onClick={handleStart} disabled={running || startingSim} variant="lime">
              <Play size={16} className="mr-1" />
              {startingSim ? "Starting..." : "Start Engine"}
            </Button>
            <Button onClick={handleStop} disabled={!running || stoppingSim} variant="destructive">
              <Square size={16} className="mr-1" />
              {stoppingSim ? "Stopping..." : "Stop Engine"}
            </Button>
            {simState && (
              <div className="ml-auto flex items-center gap-5 text-sm text-michi-muted font-medium">
                <span className="flex items-center gap-1.5"><Clock size={14} /> {simState.current_time ?? "—"}</span>
                <span className="flex items-center gap-1.5"><Users size={14} /> {simState.station_count ?? 0} stations</span>
                <Badge className={driftBadge(simState.drift_status)}>
                  Drift: {simState.drift_status.toUpperCase()}
                </Badge>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <SimulationMetrics />

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>Station-Level Simulation Data</CardTitle>
          {stationData.length > 0 && (
            <span className="text-sm text-michi-muted font-medium">Updates every 5s · {stationData.length} stations</span>
          )}
        </CardHeader>
        <CardContent>
          {!running && !simState?.running ? (
            <div className="text-center py-16">
              <Activity className="h-14 w-14 text-michi-border mx-auto mb-4" />
              <p className="text-lg font-semibold text-michi-dark">Start the simulation engine</p>
              <p className="text-sm text-michi-muted mt-1">Data refreshes every 5 seconds with predicted vs actual passenger counts</p>
            </div>
          ) : stationData.length === 0 ? (
            <div className="text-center py-10">
              <div className="animate-pulse flex flex-col items-center gap-2">
                <Activity className="h-10 w-10 text-michi-lime" />
                <p className="text-sm text-michi-muted font-medium">Waiting for simulation data...</p>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-michi-border text-left">
                    <th className="pb-3 pr-4 font-semibold text-michi-muted">Stop Name</th>
                    <th className="pb-3 pr-4 font-semibold text-michi-muted text-right">Actual Passengers</th>
                    <th className="pb-3 pr-4 font-semibold text-michi-muted text-right">Forecast</th>
                    <th className="pb-3 pr-4 font-semibold text-michi-muted text-right">Accuracy</th>
                    <th className="pb-3 pr-4 font-semibold text-michi-muted text-right">Forecast Range</th>
                    <th className="pb-3 font-semibold text-michi-muted text-right">Error Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {stationData.slice(0, 20).map((s) => (
                    <tr key={s.station_id} className="border-b border-michi-border/50 hover:bg-michi-warm transition-colors">
                      <td className="py-3 pr-4 font-semibold text-michi-dark truncate max-w-[200px]">{s.name}</td>
                      <td className="py-3 pr-4 text-right font-mono text-michi-body">{s.actual.toLocaleString()}</td>
                      <td className="py-3 pr-4 text-right font-mono text-michi-body">{s.predicted.toLocaleString()}</td>
                      <td className="py-3 pr-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-20 h-2.5 bg-michi-warm rounded-full overflow-hidden">
                            <div className="h-full bg-michi-lime rounded-full" style={{ width: `${s.confidence * 100}%` }} />
                          </div>
                          <span className="text-xs font-mono text-michi-muted">{(s.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="py-3 pr-4 text-right font-mono text-xs text-michi-muted">
                        {s.confidence_lower.toLocaleString()} – {s.confidence_upper.toLocaleString()}
                      </td>
                      <td className={`py-3 text-right font-mono font-semibold ${errorColor(s.error_pct)}`}>
                        {s.error_pct.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {stationData.length > 20 && (
                <p className="text-sm text-michi-muted mt-3 text-center font-medium">Showing top 20 of {stationData.length} stations</p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {stationData.length > 0 && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                <Users size={14} /> Total Actual Passengers
              </div>
              <p className="text-3xl font-extrabold text-michi-dark">
                {stationData.reduce((sum, s) => sum + s.actual, 0).toLocaleString()}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                <BarChart3 size={14} /> Total Forecasted
              </div>
              <p className="text-3xl font-extrabold text-michi-dark">
                {stationData.reduce((sum, s) => sum + s.predicted, 0).toLocaleString()}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                <TrendingUp size={14} /> Average Accuracy
              </div>
              <p className="text-3xl font-extrabold text-michi-lime-dark">
                {(stationData.reduce((sum, s) => sum + s.confidence, 0) / stationData.length * 100).toFixed(1)}%
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center gap-2 text-sm text-michi-muted font-medium mb-2">
                <AlertTriangle size={14} /> Stops with High Error
              </div>
              <p className={`text-3xl font-extrabold ${stationData.filter(s => s.error_pct > 15).length > 0 ? "text-michi-red" : "text-michi-lime-dark"}`}>
                {stationData.filter(s => s.error_pct > 15).length} / {stationData.length}
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}