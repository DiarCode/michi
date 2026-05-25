import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import CommandCenter from "./routes/CommandCenter";
import LiveMap from "./routes/LiveMap";
import AlertsPage from "./routes/AlertsPage";
import ScenarioPlanner from "./routes/ScenarioPlanner";
import Settings from "./routes/Settings";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex h-screen bg-gray-50">
          <aside className="w-64 bg-slate-900 text-white flex flex-col">
            <div className="p-4 text-xl font-bold">Michi</div>
            <nav className="flex-1">
              {[{to:"/",label:"Command Center"},{to:"/map",label:"Live Map"},{to:"/alerts",label:"Alerts"},{to:"/scenarios",label:"Scenarios"},{to:"/settings",label:"Settings"}].map((item)=>(
                <a key={item.to} href={item.to} className="block px-4 py-3 hover:bg-slate-800">{item.label}</a>
              ))}
            </nav>
          </aside>
          <div className="flex-1 flex flex-col">
            <header className="h-16 bg-white border-b flex items-center justify-between px-6">
              <h1 className="text-lg font-semibold">Astana Transit Intelligence</h1>
            </header>
            <main className="flex-1 overflow-auto">
              <Routes>
                <Route path="/" element={<CommandCenter />} />
                <Route path="/map" element={<LiveMap />} />
                <Route path="/alerts" element={<AlertsPage />} />
                <Route path="/scenarios" element={<ScenarioPlanner />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
