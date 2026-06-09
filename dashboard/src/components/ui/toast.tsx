import { useToasts } from "@/lib/toast";
import { X, CheckCircle, AlertTriangle, Info, AlertCircle } from "lucide-react";

const VARIANT_STYLES = {
  success: "bg-michi-lime/10 border-michi-lime/30 text-michi-lime-dark",
  error: "bg-michi-red/8 border-michi-red/30 text-michi-red",
  warning: "bg-michi-amber/10 border-michi-amber/30 text-michi-amber",
  info: "bg-michi-warm border-michi-border text-michi-body",
} as const;

const VARIANT_ICONS = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
} as const;

export function ToastContainer() {
  const { toasts, dismiss } = useToasts();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2.5 max-w-sm">
      {toasts.map((t) => {
        const Icon = VARIANT_ICONS[t.variant];
        return (
          <div
            key={t.id}
            className={`flex items-start gap-2.5 px-4 py-3.5 rounded-xl border shadow-tooltip animate-in slide-in-from-bottom-2 font-medium ${VARIANT_STYLES[t.variant]}`}
          >
            <Icon className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <p className="text-sm flex-1">{t.message}</p>
            <button
              onClick={() => dismiss(t.id)}
              className="text-current opacity-50 hover:opacity-100 flex-shrink-0"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        );
      })}
    </div>
  );
}