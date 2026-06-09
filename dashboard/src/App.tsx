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
  Calendar, Settings as SettingsIcon, ChevronDown,
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
  superadmin: "Super Admin",
};

function AppInner() {
  const [role, setRole] = useState<UserRole>(() => {
    return (localStorage.getItem("michi-role") as UserRole) || "dispatch";
  });
  const [roleMenuOpen, setRoleMenuOpen] = useState(false);

  const subscribeBuses = useBusStore((s) => s.subscribe);
  const initConnection = useConnectionStore((s) => s.init);

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
      <div className="min-h-screen bg-michi-page">
        {/* Fixed Top Bar */}
        <header className="fixed top-0 left-0 right-0 z-50 h-16 bg-white border-b border-michi-border flex items-center justify-between px-6">
          {/* Left: Logo */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-9 h-9 rounded-xl bg-michi-lime flex items-center justify-center">
              <span className="text-michi-dark font-extrabold text-sm">M</span>
            </div>
            <div>
              <div className="font-extrabold text-base text-michi-dark leading-tight">Michi</div>
              <div className="text-[10px] text-michi-muted leading-tight font-medium">Astana Transit</div>
            </div>
          </div>

          {/* Center: Pill Navigation */}
          <nav className="flex items-center gap-1.5 overflow-x-auto mx-8 scrollbar-hide">
            {nav.map(({ to, label, Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold transition-all whitespace-nowrap ${
                    isActive
                      ? "bg-michi-dark text-white shadow-sm"
                      : "text-michi-body hover:bg-michi-warm hover:text-michi-dark"
                  }`
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Right: Role Selector + Date */}
          <div className="flex items-center gap-4 shrink-0">
            <span className="text-sm text-michi-muted font-medium hidden lg:block">
              {new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
            </span>
            <div className="relative">
              <button
                onClick={() => setRoleMenuOpen(!roleMenuOpen)}
                className="flex items-center gap-2 px-3.5 py-2 rounded-full border border-michi-border bg-michi-warm hover:bg-michi-border transition-colors text-sm font-semibold text-michi-dark"
              >
                <span className="w-2 h-2 rounded-full bg-michi-lime" />
                {ROLE_LABELS[role]}
                <ChevronDown size={14} className="text-michi-muted" />
              </button>
              {roleMenuOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setRoleMenuOpen(false)} />
                  <div className="absolute right-0 top-full mt-2 w-52 bg-white rounded-2xl border border-michi-border shadow-card-hover py-2 z-50">
                    {(Object.entries(ROLE_LABELS) as [UserRole, string][]).map(([k, v]) => (
                      <button
                        key={k}
                        onClick={() => {
                          setRole(k);
                          localStorage.setItem("michi-role", k);
                          setRoleMenuOpen(false);
                        }}
                        className={`w-full text-left px-4 py-2.5 text-sm font-medium transition-colors ${
                          role === k ? "bg-michi-lime/15 text-michi-dark" : "text-michi-body hover:bg-michi-warm"
                        }`}
                      >
                        {v}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="pt-16 min-h-screen">
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