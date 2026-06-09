import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import { showToast } from "@/lib/toast";
import { TableSkeleton } from "@/components/ui/skeleton";
import { Clock, ArrowRight } from "lucide-react";

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

  if (routes.length === 0) return <div className="p-8"><TableSkeleton rows={5} /></div>;

  const filteredSchedule = schedule
    ? filterHour < 0
      ? schedule.schedule
      : schedule.schedule.filter((e) => {
          const h = parseInt(e.time.split(":")[0], 10);
          return h === filterHour;
        })
    : [];

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Timetable</h1>
        <p className="text-base text-michi-muted mt-1">Scheduled departure times for each route</p>
      </div>

      <div className="flex gap-5 items-end">
        <div className="flex-1">
          <label className="block text-sm font-semibold text-michi-dark mb-2">Route</label>
          <select
            className="w-full border border-michi-border rounded-xl px-4 py-3 bg-white text-michi-dark font-medium focus:ring-2 focus:ring-michi-lime/50 outline-none"
            value={selectedRoute}
            onChange={(e) => setSelectedRoute(e.target.value)}
          >
            {routes.map((r) => (
              <option key={r.id} value={r.id}>{r.name || r.id}</option>
            ))}
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-sm font-semibold text-michi-dark mb-2">Hour Filter</label>
          <select
            className="w-full border border-michi-border rounded-xl px-4 py-3 bg-white text-michi-dark font-medium focus:ring-2 focus:ring-michi-lime/50 outline-none"
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

      {loading && <p className="text-michi-muted font-medium">Loading schedule...</p>}

      {schedule && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
            <Card>
              <CardContent className="p-5 text-center">
                <p className="text-sm text-michi-muted font-medium uppercase">First Bus</p>
                <p className="text-3xl font-extrabold text-michi-dark mt-2">{schedule.first_bus}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5 text-center">
                <p className="text-sm text-michi-muted font-medium uppercase">Last Bus</p>
                <p className="text-3xl font-extrabold text-michi-dark mt-2">{schedule.last_bus}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5 text-center">
                <p className="text-sm text-michi-muted font-medium uppercase">Headway</p>
                <p className="text-3xl font-extrabold text-michi-dark mt-2">{schedule.headway_min} min</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-5 text-center">
                <p className="text-sm text-michi-muted font-medium uppercase">Stops</p>
                <p className="text-3xl font-extrabold text-michi-dark mt-2">{schedule.stops.length}</p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader><CardTitle>{schedule.route_name} Schedule</CardTitle></CardHeader>
            <CardContent>
              {filteredSchedule.length === 0 ? (
                <div className="text-center py-10">
                  <Clock size={28} className="text-michi-border mx-auto mb-3" />
                  <p className="text-base text-michi-muted font-medium">No departures for this hour</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-michi-border">
                        <th className="text-left py-2.5 font-semibold text-michi-muted">Time</th>
                        <th className="text-left py-2.5 font-semibold text-michi-muted">Stop</th>
                        <th className="text-right py-2.5 font-semibold text-michi-muted">Headway</th>
                        <th className="text-left py-2.5 font-semibold text-michi-muted">Direction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredSchedule.map((entry, i) => (
                        <tr key={i} className="border-b border-michi-border/50 hover:bg-michi-warm transition-colors">
                          <td className="py-2.5 font-mono font-semibold text-michi-dark">{entry.time}</td>
                          <td className="py-2.5 font-semibold text-michi-dark">{entry.stop_name}</td>
                          <td className="text-right py-2.5 text-michi-body">{entry.headway_min} min</td>
                          <td className="py-2.5">
                            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${entry.direction === "outbound" ? "bg-michi-lime/15 text-michi-lime-dark" : "bg-michi-teal/15 text-michi-teal"}`}>
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
                    <span className="bg-michi-lime/15 text-michi-lime-dark px-4 py-2 rounded-full text-sm font-semibold">{stop.name}</span>
                    {i < schedule.stops.length - 1 && (
                      <ArrowRight size={16} className="text-michi-muted" />
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