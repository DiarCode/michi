import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import type { UserRole } from "@/types";
import { ROLE_LABELS } from "@/App";
import { HugeiconsIcon } from "@hugeicons/react";
import { PaintBrush01Icon, UserIcon, Sun01Icon, Moon02Icon, DashboardCircleIcon } from "@/lib/icons";
import { useThemeStore, type Theme } from "@/stores/themeStore";

const THEME_OPTIONS: { value: Theme; label: string; Icon: any }[] = [
  { value: "light", label: "Light", Icon: Sun01Icon },
  { value: "dark", label: "Dark", Icon: Moon02Icon },
  { value: "system", label: "System", Icon: DashboardCircleIcon },
];

export default function Settings() {
  const currentRole = (localStorage.getItem("michi-role") || "dispatch") as UserRole;
  const { theme, resolvedTheme, setTheme } = useThemeStore();

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-foreground">Settings</h1>
        <p className="text-base text-muted-foreground mt-1">Application preferences and role configuration</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HugeiconsIcon icon={PaintBrush01Icon} size={18} className="text-chart-2" />
            Appearance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-3">
              {THEME_OPTIONS.map(({ value, label, Icon }) => (
                <button
                  key={value}
                  onClick={() => setTheme(value)}
                  className={`flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all ${
                    theme === value
                      ? "border-primary bg-primary/10 shadow-sm"
                      : "border-border hover:border-muted-foreground bg-card"
                  }`}
                >
                  <HugeiconsIcon icon={Icon} size={22} className={theme === value ? "text-primary" : "text-muted-foreground"} />
                  <span className={`text-sm font-semibold ${theme === value ? "text-foreground" : "text-muted-foreground"}`}>
                    {label}
                  </span>
                </button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              {resolvedTheme === "dark"
                ? "Dark mode is active — map tiles and UI adapt automatically."
                : "Light mode is active."}
              {theme === "system" && " (Following your system preference)"}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HugeiconsIcon icon={UserIcon} size={18} className="text-chart-2" />
            Role Selection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground font-medium">Select your role to customize navigation and available features</p>
          <div className="flex flex-wrap gap-2">
            {(Object.entries(ROLE_LABELS) as [UserRole, string][]).map(([k, v]) => (
              <button
                key={k}
                onClick={() => { localStorage.setItem("michi-role", k); window.location.reload(); }}
                className={`px-4 py-2 text-sm rounded-full font-semibold transition-all ${
                  currentRole === k
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "bg-muted text-muted-foreground border border-border hover:bg-border"
                }`}
              >
                {v}
              </button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground font-medium">Changing role will reload the page</p>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-semibold text-foreground">Version</p>
              <p className="text-sm text-muted-foreground">Michi Dashboard v2.0</p>
            </div>
            <span className="text-xs text-muted-foreground font-medium">© {new Date().getFullYear()}</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}