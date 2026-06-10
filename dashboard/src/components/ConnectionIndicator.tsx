import { useConnectionStore } from "@/stores/connectionStore";

const stateConfig: Record<string, { dotClass: string; label: string; pulse: boolean }> = {
  connected: { dotClass: "bg-chart-2", label: "Live", pulse: false },
  connecting: { dotClass: "bg-chart-4", label: "Connecting", pulse: true },
  disconnected: { dotClass: "bg-destructive", label: "Offline", pulse: true },
};

export function ConnectionIndicator() {
  const wsState = useConnectionStore((s) => s.wsState);
  const config = stateConfig[wsState] ?? stateConfig.disconnected;

  return (
    <div
      className="flex items-center gap-1.5 text-xs text-muted-foreground font-medium"
      title={`WebSocket: ${wsState}`}
      data-testid="connection-indicator"
      data-state={wsState}
    >
      <span className="relative flex h-2 w-2">
        {config.pulse && (
          <span
            className={`absolute inline-flex h-full w-full animate-ping rounded-full ${config.dotClass} opacity-60`}
          />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${config.dotClass}`} />
      </span>
      <span>{config.label}</span>
    </div>
  );
}
