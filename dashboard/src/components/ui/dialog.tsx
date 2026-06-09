import { type ReactNode, useEffect, useRef } from "react";

interface Props { open: boolean; onClose: () => void; children: ReactNode; title?: string }

export function Dialog({ open, onClose, children, title }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { if (open) ref.current?.showModal(); else ref.current?.close(); }, [open]);
  if (!open) return null;
  return (
    <dialog ref={ref} className="rounded-2xl p-0 shadow-tooltip border border-michi-border backdrop:bg-black/30" onClose={onClose}>
      <div className="p-6 bg-white">
        {title && <h2 className="text-lg font-bold text-michi-dark mb-4">{title}</h2>}
        {children}
      </div>
    </dialog>
  );
}