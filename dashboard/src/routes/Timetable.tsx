import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import { showToast } from "@/lib/toast";
import { TableSkeleton } from "@/components/ui/skeleton";

interface ScheduleEntry {
  stop_id: string;
  stop_name: string;
  time: string;
  headway_min: number;
  direction: string;
}

interface RouteSchedule {
  route_id: string;
  route_name: string;
  stops: { id: string; name: string }[];
  schedule: ScheduleEntry[];
  first_bus: string;
  last_bus: string;
  headway_min: number;
}

export default function Timetable() {
  const [routes, setRoutes] = useState<{ id: string; name: string }[]>([]);
  const [selectedRoute, setSelectedRoute] = useState<string>("");
  const [schedule, setSchedule] = useState<RouteSchedule | null>(null);
  const [loading, setLoading] = useState(false);
  const [filterHour, setFilterHour] = useState<number>(-1);

  useEffect(() => {
    api.get("/routes").then(({ data }) => {
      const r = data.routes || data;
      setRoutes(r);
      if (r.length > 0 && !selectedRoute) setSelectedRoute(r[0].id);
    });
  }, []);

  useEffect(() => {
    if (!selectedRoute) return;
    setLoading(true);
    api.get("/routes/" + selectedRoute + "/schedule")
      .then(({ data }) => setSchedule(data))
      .catch((err) => { showToast.error(`Failed to load schedule: ${err.message}`); setSchedule(null); })
      .finally(() => setLoading(false));
  }, [selectedRoute]);

  if (routes.length === 0) return <div className="p-6"><TableSkeleton rows={5} /></div>;

  const filteredSchedule = schedule
    ? filterHour < 0
      ? schedule.schedule
      : schedule.schedule.filter((e) => {
          const h = parseInt(e.time.split(":")[0], 10);
          return h === filterHour;
        })
    : [];

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Timetable</h2>
      <p className="text-sm text-gray-500">View scheduled departure times for each route.</p>

      <div className="flex gap-4 items-end">
        <div className="flex-1">
          <label className="block text-sm font-medium mb-1">Route</label>
          <select
            className="w-full border rounded px-3 py-2"
            value={selectedRoute}
            onChange={(e) => setSelectedRoute(e.target.value)}
          >
            {routes.map((r) => (
              <option key={r.id} value={r.id}>{r.name || r.id}</option>
            ))}
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium mb-1">Hour Filter</label>
          <select
            className="w-full border rounded px-3 py-2"
            value={filterHour}
            onChange={(e) => setFilterHour(Number(e.target.value))}
          >
            <option value={-1}>All Hours</option>
            {Array.from({ length: 24 }, (_, i) => (
              <option key={i} value={i}>{String(i).padStart(2, "0")}:00–{String(i).padStart(2, "0")}:59</option>
            ))}
          </select>
        </div>
      </div>

      {loading && <p className="text-gray-500">Loading schedule...</p>}

      {schedule && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-xs text-gray-500 uppercase">First Bus</p>
                <p className="text-xl font-bold">{schedule.first_bus}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-xs text-gray-500 uppercase">Last Bus</p>
                <p className="text-xl font-bold">{schedule.last_bus}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-xs text-gray-500 uppercase">Headway</p>
                <p className="text-xl font-bold">{schedule.headway_min} min</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-xs text-gray-500 uppercase">Stops</p>
                <p className="text-xl font-bold">{schedule.stops.length}</p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader><CardTitle>{schedule.route_name} Schedule</CardTitle></CardHeader>
            <CardContent>
              {filteredSchedule.length === 0 ? (
                <p className="text-gray-500 text-sm">No departures for this hour.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2">Time</th>
                        <th className="text-left py-2">Stop</th>
                        <th className="text-right py-2">Headway</th>
                        <th className="text-left py-2">Direction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredSchedule.map((entry, i) => (
                        <tr key={i} className="border-b hover:bg-gray-50">
                          <td className="py-2 font-mono font-medium">{entry.time}</td>
                          <td className="py-2">{entry.stop_name}</td>
                          <td className="text-right py-2">{entry.headway_min} min</td>
                          <td className="py-2">
                            <span className={entry.direction === "outbound" ? "text-blue-600" : "text-green-600"}>
                              {entry.direction}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Route Stops</CardTitle></CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 flex-wrap">
                {schedule.stops.map((stop, i) => (
                  <div key={stop.id} className="flex items-center gap-2">
                    <span className="bg-blue-100 text-blue-800 px-3 py-1.5 rounded-full text-sm font-medium">{stop.name}</span>
                    {i < schedule.stops.length - 1 && (
                      <span className="text-gray-400">→</span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}