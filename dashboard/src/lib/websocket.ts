const WS_BASE = (import.meta.env.VITE_WS_URL as string) || "ws://localhost:8000/ws";

export type WSEvent = {
  type: "bus_position" | "alert" | "forecast_update";
  data: Record<string, unknown>;
};

type Listener = (event: WSEvent) => void;

export class WSClient {
  private ws: WebSocket | null = null;
  private listeners: Listener[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  connect() {
    this.ws = new WebSocket(WS_BASE + "/realtime");
    this.ws.onmessage = (e) => {
      try {
        const event: WSEvent = JSON.parse(e.data);
        this.listeners.forEach((fn) => fn(event));
      } catch { /* ignore malformed */ }
    };
    this.ws.onclose = () => {
      this.reconnectTimer = setTimeout(() => this.connect(), 3000);
    };
  }

  subscribe(fn: Listener) {
    this.listeners.push(fn);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== fn);
    };
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
  }
}

export const wsClient = new WSClient();
