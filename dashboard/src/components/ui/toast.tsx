import { useToasts } from "@/lib/toast";
import { HugeiconsIcon } from "@hugeicons/react";
import { Cancel01Icon, CheckmarkCircle01Icon, Alert01Icon, InformationCircleIcon, AlertCircleIcon } from "@/lib/icons";

const VARIANT_STYLES = {
  success: "bg-chart-2/10 border-chart-2/30 text-chart-2",
  error: "bg-destructive/10 border-destructive/30 text-destructive",
  warning: "bg-chart-4/10 border-chart-4/30 text-chart-4",
  info: "bg-muted border-border text-muted-foreground",
} as const;

const VARIANT_ICONS = {
  success: CheckmarkCircle01Icon,
  error: AlertCircleIcon,
  warning: Alert01Icon,
  info: InformationCircleIcon,
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
            className={`flex items-start gap-2.5 px-4 py-3.5 rounded-xl border shadow-lg animate-in slide-in-from-bottom-2 font-medium ${VARIANT_STYLES[t.variant]}`}
          >
            <HugeiconsIcon icon={Icon} className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <p className="text-sm flex-1">{t.message}</p>
            <button
              onClick={() => dismiss(t.id)}
              className="text-current opacity-50 hover:opacity-100 flex-shrink-0"
            >
              <HugeiconsIcon icon={Cancel01Icon} className="h-4 w-4" />
            </button>
          </div>
        );
      })}
    </div>
  );
}