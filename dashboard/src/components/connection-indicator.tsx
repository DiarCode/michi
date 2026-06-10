import { useEffect, useState } from "react"
import { useConnectionStore } from "@/stores/connection-store"
import { cn } from "@/lib/utils"

const LABEL: Record<string, string> = {
  open: "Live",
  connecting: "Connecting",
  closed: "Offline",
  error: "Error",
}

const COLOR: Record<string, string> = {
  open: "bg-emerald-500",
  connecting: "bg-amber-500",
  closed: "bg-zinc-400",
  error: "bg-rose-500",
}

export function ConnectionIndicator() {
  const connected = useConnectionStore((s) => s.connected)
  const lastTickReceived = useConnectionStore((s) => s.lastTickReceived)
  const [now, setNow] = useState(Date.now())

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 30_000)
    return () => clearInterval(t)
  }, [])

  const status = connected ? "open" : "closed"

  return (
    <div
      title={`Last tick #${lastTickReceived} · ${new Date(now).toLocaleTimeString()}`}
      className="flex items-center gap-2 rounded-2xl bg-muted/50 px-2 py-1 text-xs"
    >
      <span className={cn("size-1.5 rounded-full", COLOR[status])} />
      <span className="text-muted-foreground">{LABEL[status]}</span>
    </div>
  )
}
