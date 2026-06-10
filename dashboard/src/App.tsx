import { Route, Routes } from "react-router-dom"

import { AppShell } from "@/components/app-shell"
import { CommandCenterPage } from "@/routes/command-center"
import { LiveMapPage } from "@/routes/live-map"
import { AlertsPage } from "@/routes/alerts"
import { SimulationPage } from "@/routes/simulation"
import { ForecastPage } from "@/routes/forecast"
import { ExecutivePage } from "@/routes/executive"
import { SettingsPage } from "@/routes/settings"

function NotFound() {
  return (
    <div className="grid place-items-center py-32 text-center">
      <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">404</p>
      <h1 className="mt-2 font-heading text-3xl font-medium">Page not found</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        The route you tried doesn&apos;t exist. Press <kbd className="rounded border px-1">⌘</kbd>
        <kbd className="rounded border px-1">K</kbd> to open the command palette.
      </p>
    </div>
  )
}

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<CommandCenterPage />} />
        <Route path="/map" element={<LiveMapPage />} />
        <Route path="/alerts" element={<AlertsPage />} />
        <Route path="/simulation" element={<SimulationPage />} />
        <Route path="/forecast" element={<ForecastPage />} />
        <Route path="/executive" element={<ExecutivePage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </AppShell>
  )
}

export default App
