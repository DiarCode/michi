import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import CommandCenter from "./routes/CommandCenter";
import LiveMap from "./routes/LiveMap";
import AlertsPage from "./routes/AlertsPage";
import Settings from "./routes/Settings";
import SimulationPage from "./routes/SimulationPage";
import ForecastPage from "./routes/ForecastPage";
import ExecutivePage from "./routes/ExecutivePage";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  BarChartIcon, DashboardCircleIcon, Alert01Icon,
  ActivityIcon,
  Settings01Icon, ChevronDownIcon,
} from "@/lib/icons";
import type { UserRole } from "./types";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ToastContainer } from "./components/ui/toast";
import { ConnectionIndicator } from "./components/ConnectionIndicator";
import { useBusStore } from "./stores/busStore";
import { useConnectionStore } from "./stores/connectionStore";
import { useThemeStore } from "./stores/themeStore";

const queryClient = new QueryClient();

type NavItem = { to: string; label: string; Icon: any };

const ROLE_NAV: Record<UserRole, NavItem[]> = {
  dispatch: [
    { to: "/", label: "Command Center", Icon: BarChartIcon },
    { to: "/map", label: "Live Map", Icon: DashboardCircleIcon },
    { to: "/alerts", label: "Alerts", Icon: Alert01Icon },
    { to: "/settings", label: "Settings", Icon: Settings01Icon },
  ],
  research: [
    { to: "/simulation", label: "Simulation", Icon: ActivityIcon },
    { to: "/forecast", label: "Forecast", Icon: BarChartIcon },
    { to: "/settings", label: "Settings", Icon: Settings01Icon },
  ],
  planning: [
    { to: "/forecast", label: "Forecast", Icon: BarChartIcon },
    { to: "/settings", label: "Settings", Icon: Settings01Icon },
  ],
  executive: [
    { to: "/executive", label: "Executive", Icon: BarChartIcon },
    { to: "/settings", label: "Settings", Icon: Settings01Icon },
  ],
  superadmin: [
    { to: "/", label: "Command Center", Icon: BarChartIcon },
    { to: "/map", label: "Live Map", Icon: DashboardCircleIcon },
    { to: "/alerts", label: "Alerts", Icon: Alert01Icon },
    { to: "/simulation", label: "Simulation", Icon: ActivityIcon },
    { to: "/forecast", label: "Forecast", Icon: BarChartIcon },
    { to: "/executive", label: "Executive", Icon: BarChartIcon },
    { to: "/settings", label: "Settings", Icon: Settings01Icon },
  ],
};

export const ROLE_LABELS: Record<UserRole, string> = {
  dispatch: "Dispatch",
  research: "Research",
  planning: "Planning",
  executive: "Executive",
  superadmin: "Super Admin",
};

function AppInner() {
  const [role, setRole] = useState<UserRole>(() => {
    return (localStorage.getItem("michi-role") as UserRole) || "dispatch";
  });
  const [roleMenuOpen, setRoleMenuOpen] = useState(false);

  const subscribeBuses = useBusStore((s) => s.subscribe);
  const initConnection = useConnectionStore((s) => s.init);
  const setTheme = useThemeStore((s) => s.setTheme);
  const theme = useThemeStore((s) => s.theme);

  // Apply theme class on mount and when theme changes
  useEffect(() => {
    setTheme(theme);
  }, [theme, setTheme]);

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
      <div className="min-h-screen bg-background">
        {/* Fixed Top Bar */}
        <header className="fixed top-0 left-0 right-0 z-50 h-16 bg-card flex items-center justify-between px-6 border-b border-border">
          {/* Left: Logo */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
              <span className="text-primary-foreground font-extrabold text-sm">M</span>
            </div>
            <div>
              <div className="font-extrabold text-base text-foreground leading-tight">Michi</div>
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
                      ? "bg-primary/15 text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`
                }
              >
                <HugeiconsIcon icon={Icon} size={16} />
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Right: Connection + Role Selector */}
          <div className="flex items-center gap-4 shrink-0">
            <ConnectionIndicator />
            <div className="relative">
              <button
                onClick={() => setRoleMenuOpen(!roleMenuOpen)}
                className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-muted hover:bg-border transition-colors text-sm font-semibold text-foreground"
              >
                <span className="w-2 h-2 rounded-full bg-chart-2" />
                {ROLE_LABELS[role]}
                <HugeiconsIcon icon={ChevronDownIcon} size={14} className="text-muted-foreground" />
              </button>
              {roleMenuOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setRoleMenuOpen(false)} />
                  <div className="absolute right-0 top-full mt-2 w-52 bg-card rounded-2xl shadow-md py-2 z-50">
                    {(Object.entries(ROLE_LABELS) as [UserRole, string][]).map(([k, v]) => (
                      <button
                        key={k}
                        onClick={() => {
                          setRole(k);
                          localStorage.setItem("michi-role", k);
                          setRoleMenuOpen(false);
                        }}
                        className={`w-full text-left px-4 py-2.5 text-sm font-medium transition-colors ${
                          role === k ? "bg-primary/15 text-primary" : "text-muted-foreground hover:bg-muted"
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
            <Route path="/forecast" element={<ForecastPage />} />
            <Route path="/executive" element={<ExecutivePage />} />
            <Route path="/settings" element={<Settings />} />
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