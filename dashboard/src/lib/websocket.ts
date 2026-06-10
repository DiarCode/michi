// Resolve the WebSocket base URL. When unset (production in
// docker-compose behind the nginx sidecar) we use a relative path so the
// browser opens ws(s)://<current-host>/ws/realtime and the proxy forwards
// it to the FastAPI backend.
const WS_BASE =
  (import.meta.env.VITE_WS_URL as string | undefined)?.trim() ||
  (typeof window !== "undefined"
    ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`
    : "ws://localhost:8000/ws");

export type WSEventType = "bus_position" | "alert" | "forecast_update" | "simulation_tick" | "validation_metric" | "drift_alert";

export type WSEvent = {
  type: WSEventType;
  data: Record<string, unknown>;
};

type Listener = (event: WSEvent) => void;

const BASE_DELAY_MS = 3000;
const MAX_DELAY_MS = 30000;
const MULTIPLIER = 1.5;

export class WSClient {
  private ws: WebSocket | null = null;
  private listeners: Listener[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private currentDelayMs = BASE_DELAY_MS;
  private lastTickReceived = 0;
  private subscriptions: WSEventType[] | null = null;

  /** Set event type subscriptions. Call before connect or to update after. */
  subscribeTo(types: WSEventType[]) {
    this.subscriptions = types;
    // If already connected, send the updated subscription list
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ subscribe: types }));
    }
  }

  connect() {
    this.ws = new WebSocket(WS_BASE + "/realtime");
    this.ws.onopen = () => {
      // Reset backoff on successful connection
      this.reconnectAttempts = 0;
      this.currentDelayMs = BASE_DELAY_MS;

      // State catch-up: send last tick received so server can resend missed events
      if (this.lastTickReceived > 0) {
        this.ws!.send(JSON.stringify({ last_tick: this.lastTickReceived }));
      }

      // Send subscription filter if set
      if (this.subscriptions) {
        this.ws!.send(JSON.stringify({ subscribe: this.subscriptions }));
      }
    };
    this.ws.onmessage = (e) => {
      try {
        const event: WSEvent = JSON.parse(e.data);
        // Track simulation tick counter for state catch-up
        if (event.type === "simulation_tick" && typeof event.data?.tick === "number") {
          this.lastTickReceived = event.data.tick as number;
        }
        this.listeners.forEach((fn) => fn(event));
      } catch {
        /* ignore malformed */
      }
    };
    this.ws.onclose = () => {
      this.scheduleReconnect();
    };
    this.ws.onerror = () => {
      // onclose will fire after onerror, which handles reconnect
    };
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return; // already scheduled

    // Exponential backoff with jitter
    const jitter = Math.random() * 1000; // 0-1s random jitter
    const delay = Math.min(this.currentDelayMs + jitter, MAX_DELAY_MS);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.reconnectAttempts += 1;
      this.currentDelayMs = Math.min(this.currentDelayMs * MULTIPLIER, MAX_DELAY_MS);
      this.connect();
    }, delay);
  }

  subscribe(fn: Listener) {
    this.listeners.push(fn);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== fn);
    };
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.ws?.close();
  }
}

export const wsClient = new WSClient();