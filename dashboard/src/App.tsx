import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import CommandCenter from "./routes/CommandCenter";
import LiveMap from "./routes/LiveMap";
import AlertsPage from "./routes/AlertsPage";
import Settings from "./routes/Settings";
import Reports from "./routes/Reports";
import Timetable from "./routes/Timetable";
import TrainingPage from "./routes/TrainingPage";
import AnalyticsPage from "./routes/AnalyticsPage";
import SimulationPage from "./routes/SimulationPage";
import NetworkPage from "./routes/NetworkPage";
import ForecastPage from "./routes/ForecastPage";
import ExecutivePage from "./routes/ExecutivePage";
import DepotPage from "./routes/DepotPage";
import PassengerPage from "./routes/PassengerPage";
import {
  BarChart, Map, AlertTriangle, FileText,
  Activity, TrendingUp, GitGraph, BrainCircuit,
  BarChart3, Truck, Users,
  Calendar, Settings as SettingsIcon,
} from "lucide-react";
import type { UserRole } from "./types";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ToastContainer } from "./components/ui/toast";
import { useBusStore } from "./stores/busStore";
import { useConnectionStore } from "./stores/connectionStore";

const queryClient = new QueryClient();

type NavItem = { to: string; label: string; Icon: React.ComponentType<{ size?: number; className?: string }> };

const ROLE_NAV: Record<UserRole, NavItem[]> = {
  dispatch: [
    { to: "/", label: "Command Center", Icon: BarChart },
    { to: "/map", label: "Live Map", Icon: Map },
    { to: "/alerts", label: "Alerts", Icon: AlertTriangle },
    { to: "/settings", label: "Settings", Icon: SettingsIcon },
  ],
  research: [
    { to: "/training", label: "Training", Icon: BrainCircuit },
    { to: "/forecast", label: "Forecast", Icon: BarChart3 },
    { to: "/analytics", label: "Analytics", Icon: TrendingUp },
    { to: "/simulation", label: "Simulation", Icon: Activity },
    { to: "/settings", label: "Settings", Icon: SettingsIcon },
  ],
  planning: [
    { to: "/forecast", label: "Forecast", Icon: BarChart3 },
    { to: "/analytics", label: "Analytics", Icon: TrendingUp },
    { to: "/network", label: "Network", Icon: GitGraph },
    { to: "/reports", label: "Reports", Icon: FileText },
    { to: "/settings", label: "Settings", Icon: SettingsIcon },
  ],
  executive: [
    { to: "/executive", label: "Executive Dashboard", Icon: BarChart3 },
    { to: "/reports", label: "Reports", Icon: FileText },
    { to: "/settings", label: "Settings", Icon: SettingsIcon },
  ],
  depot: [
    { to: "/depot", label: "Depot Operations", Icon: Truck },
    { to: "/alerts", label: "Alerts", Icon: AlertTriangle },
    { to: "/settings", label: "Settings", Icon: SettingsIcon },
  ],
  passenger: [
    { to: "/passenger", label: "Passenger Info", Icon: Users },
    { to: "/timetable", label: "Timetable", Icon: Calendar },
    { to: "/map", label: "Live Map", Icon: Map },
    { to: "/settings", label: "Settings", Icon: SettingsIcon },
  ],
  superadmin: [
    { to: "/", label: "Command Center", Icon: BarChart },
    { to: "/map", label: "Live Map", Icon: Map },
    { to: "/alerts", label: "Alerts", Icon: AlertTriangle },
    { to: "/simulation", label: "Simulation", Icon: Activity },
    { to: "/analytics", label: "Analytics", Icon: TrendingUp },
    { to: "/network", label: "Network", Icon: GitGraph },
    { to: "/forecast", label: "Forecast", Icon: BarChart3 },
    { to: "/training", label: "Training", Icon: BrainCircuit },
    { to: "/executive", label: "Executive Dashboard", Icon: BarChart3 },
    { to: "/depot", label: "Depot Operations", Icon: Truck },
    { to: "/passenger", label: "Passenger Info", Icon: Users },
    { to: "/timetable", label: "Timetable", Icon: Calendar },
    { to: "/reports", label: "Reports", Icon: FileText },
    { to: "/settings", label: "Settings", Icon: SettingsIcon },
  ],
};

export const ROLE_LABELS: Record<UserRole, string> = {
  dispatch: "Dispatch",
  research: "Research",
  planning: "Planning",
  executive: "Executive",
  depot: "Depot",
  passenger: "Passenger",
  superadmin: "Super Admin (All Access)",
};

function AppInner() {
  const [role, setRole] = useState<UserRole>(() => {
    return (localStorage.getItem("michi-role") as UserRole) || "dispatch";
  });

  const subscribeBuses = useBusStore((s) => s.subscribe);
  const initConnection = useConnectionStore((s) => s.init);

  // Initialize WS subscriptions at app root
  useEffect(() => {
    const unsubBuses = subscribeBuses();
    const cleanupConn = initConnection();
    return () => {
      unsubBuses();
      if (typeof cleanupConn === "function") cleanupConn();
    };
  }, [subscribeBuses, initConnection]);

  const nav = ROLE_NAV[role];

  return (
    <BrowserRouter>
        <div className="flex h-screen bg-gray-50 dark:bg-gray-950">
          <aside className="w-64 bg-slate-900 dark:bg-slate-950 text-white flex flex-col border-r border-slate-700/50">
            <div className="p-4 border-b border-slate-700/50">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-sm">M</div>
                <div>
                  <div className="font-bold text-sm leading-tight">Michi</div>
                  <div className="text-[10px] text-slate-400 leading-tight">Astana Transit</div>
                </div>
              </div>
            </div>
            <div className="px-3 py-2 border-b border-slate-700/50">
              <p className="text-[10px] text-slate-400 mb-1">Role</p>
              <select
                value={role}
                onChange={(e) => { setRole(e.target.value as UserRole); localStorage.setItem("michi-role", e.target.value); }}
                className="w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-600"
              >
                {Object.entries(ROLE_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>{v}</option>
                ))}
              </select>
            </div>
            <nav className="flex-1 py-2">
              {nav.map(({ to, label, Icon }) => (
                <NavLink key={to} to={to} className={({ isActive }) => `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${isActive ? "bg-blue-600/20 text-blue-400 border-l-2 border-blue-400" : "text-slate-300 hover:bg-slate-800 hover:text-white"}`}>
                  <Icon size={18} />
                  {label}
                </NavLink>
              ))}
            </nav>
            <div className="p-3 border-t border-slate-700/50 text-[10px] text-slate-500">v2.0 · {new Date().getFullYear()}</div>
          </aside>
          <div className="flex-1 flex flex-col">
            <header className="h-14 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-6">
              <h1 className="text-base font-semibold text-gray-800 dark:text-gray-100">
                Astana Transit Intelligence — {ROLE_LABELS[role]}
              </h1>
              <span className="text-xs text-gray-400">{new Date().toLocaleDateString()}</span>
            </header>
            <main className="flex-1 overflow-auto">
              <Routes>
                <Route path="/" element={<CommandCenter />} />
                <Route path="/map" element={<LiveMap />} />
                <Route path="/alerts" element={<AlertsPage />} />
                <Route path="/simulation" element={<SimulationPage />} />
                <Route path="/analytics" element={<AnalyticsPage />} />
                <Route path="/network" element={<NetworkPage />} />
                <Route path="/forecast" element={<ForecastPage />} />
                <Route path="/training" element={<TrainingPage />} />
                <Route path="/reports" element={<Reports />} />
                <Route path="/timetable" element={<Timetable />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="/route-command" element={<Navigate to="/" replace />} />
                <Route path="/executive" element={<ExecutivePage />} />
                <Route path="/depot" element={<DepotPage />} />
                <Route path="/passenger" element={<PassengerPage />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <AppInner />
      </ErrorBoundary>
      <ToastContainer />
    </QueryClientProvider>
  );
}
