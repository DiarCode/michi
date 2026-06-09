// Light-mode only — no state toggle needed
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import type { UserRole } from "@/types";
import { ROLE_LABELS } from "@/App";
import { Palette, User } from "lucide-react";

export default function Settings() {
  const currentRole = (localStorage.getItem("michi-role") || "dispatch") as UserRole;

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Settings</h1>
        <p className="text-base text-michi-muted mt-1">Application preferences and role configuration</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Palette size={18} className="text-michi-lime-dark" />
            Appearance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between p-4 bg-michi-warm rounded-xl">
            <div>
              <p className="font-semibold text-michi-dark">Light Mode</p>
              <p className="text-sm text-michi-muted">The dashboard is optimized for light mode viewing</p>
            </div>
            <div className="w-12 h-6 rounded-full bg-michi-lime flex items-center justify-end px-0.5">
              <span className="w-5 h-5 bg-white rounded-full shadow-sm" />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User size={18} className="text-michi-lime-dark" />
            Role Selection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-michi-muted font-medium">Select your role to customize navigation and available features</p>
          <div className="flex flex-wrap gap-2">
            {(Object.entries(ROLE_LABELS) as [UserRole, string][]).map(([k, v]) => (
              <button
                key={k}
                onClick={() => { localStorage.setItem("michi-role", k); window.location.reload(); }}
                className={`px-4 py-2 text-sm rounded-full font-semibold transition-all ${
                  currentRole === k
                    ? "bg-michi-dark text-white shadow-sm"
                    : "bg-michi-warm text-michi-body border border-michi-border hover:bg-michi-border"
                }`}
              >
                {v}
              </button>
            ))}
          </div>
          <p className="text-xs text-michi-muted font-medium">Changing role will reload the page</p>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-semibold text-michi-dark">Version</p>
              <p className="text-sm text-michi-muted">Michi Dashboard v2.0</p>
            </div>
            <span className="text-xs text-michi-muted font-medium">© {new Date().getFullYear()}</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}