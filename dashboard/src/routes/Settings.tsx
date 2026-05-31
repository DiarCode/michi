import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import type { UserRole } from "@/types";
import { ROLE_LABELS } from "@/App";

export default function Settings() {
  const currentRole = (localStorage.getItem("michi-role") || "dispatch") as UserRole;
  const [dark, setDark] = useState(() => localStorage.getItem("michi_dark") === "true");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("michi_dark", String(dark));
  }, [dark]);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Settings</h2>

      <Card>
        <CardHeader><CardTitle>Appearance</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Dark Mode</p>
              <p className="text-sm text-gray-500">Switch between light and dark theme.</p>
            </div>
            <button
              onClick={() => setDark(!dark)}
              className={"relative w-12 h-6 rounded-full transition-colors " + (dark ? "bg-blue-600" : "bg-gray-300")}
            >
              <span
                className={"absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform " + (dark ? "translate-x-6" : "translate-x-0")}
              />
            </button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Role Selection</CardTitle></CardHeader>
        <CardContent>
          <select className="w-full border rounded px-3 py-2" value={currentRole} onChange={(e) => { localStorage.setItem("michi-role", e.target.value); window.location.reload(); }}>
            {(Object.entries(ROLE_LABELS) as [UserRole, string][]).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <p className="text-xs text-gray-500 mt-2">Changing role will reload the page.</p>
        </CardContent>
      </Card>
    </div>
  );
}