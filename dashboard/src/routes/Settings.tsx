import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ROLES, type Role } from "@/components/layout/RoleGuard";

export default function Settings() {
  const currentRole = localStorage.getItem("michi_role") || "Dispatch Manager";

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Settings</h2>
      <Card>
        <CardHeader><CardTitle>Role Selection</CardTitle></CardHeader>
        <CardContent>
          <select className="w-full border rounded px-3 py-2" value={currentRole} onChange={(e) => { localStorage.setItem("michi_role", e.target.value); window.location.reload(); }}>
            {ROLES.map((r: Role) => <option key={r} value={r}>{r}</option>)}
          </select>
          <p className="text-xs text-gray-500 mt-2">Changing role will reload the page.</p>
        </CardContent>
      </Card>
    </div>
  );
}
