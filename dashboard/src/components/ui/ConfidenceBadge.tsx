import { useState } from "react";

type ConfidenceLevel = "high" | "medium" | "low";

function classifyConfidence(confidence: number): ConfidenceLevel {
  if (confidence >= 0.85) return "high";
  if (confidence >= 0.65) return "medium";
  return "low";
}

const levelConfig: Record<ConfidenceLevel, { bg: string; icon: string; label: string; pulseClass: string }> = {
  high: {
    bg: "bg-chart-2/10 text-chart-2 border border-chart-2/30",
    icon: "✦",
    label: "High confidence",
    pulseClass: "",
  },
  medium: {
    bg: "bg-chart-4/10 text-chart-4 border border-chart-4/30",
    icon: "⚠",
    label: "Moderate confidence",
    pulseClass: "",
  },
  low: {
    bg: "bg-destructive/10 text-destructive border border-destructive/30",
    icon: "✗",
    label: "Low confidence",
    pulseClass: "animate-pulse",
  },
};

interface ConfidenceBadgeProps {
  /** Confidence value 0–1 */
  confidence: number;
  /** Optional model version to show in tooltip */
  modelVersion?: string;
  /** Optional deviation text (e.g. "±12%") */
  deviation?: string;
  /** Compact mode — just the icon + percentage, no expand on click */
  compact?: boolean;
  /** Extra CSS classes */
  className?: string;
}

/**
 * Crystal Ball Confidence Badge — shows model prediction confidence
 * with color-coded levels (green ≥85%, amber 65-85%, red <65%),
 * expandable tooltip with model version and deviation.
 */
export default function ConfidenceBadge({
  confidence,
  modelVersion,
  deviation,
  compact = false,
  className = "",
}: ConfidenceBadgeProps) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(confidence * 100);
  const level = classifyConfidence(confidence);
  const config = levelConfig[level];

  if (compact) {
    return (
      <span
        className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-bold font-mono ${config.bg} ${config.pulseClass} ${className}`}
        title={`${config.label}: ${pct}%${modelVersion ? ` (v${modelVersion})` : ""}`}
      >
        <span>{config.icon}</span>
        <span>{pct}%</span>
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={() => setExpanded(!expanded)}
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-semibold ${config.bg} ${config.pulseClass} cursor-pointer select-none ${className}`}
      title="Click for details"
    >
      <span className="text-sm">{config.icon}</span>
      <span className="font-mono">{pct}%</span>
      <span className="text-[10px] opacity-70">{config.label.split(" ")[0]}</span>

      {expanded && (
        <span className="ml-1.5 flex items-center gap-2 text-[10px] font-normal opacity-80">
          {modelVersion && (
            <span>v{modelVersion}</span>
          )}
          {deviation && (
            <span>{deviation}</span>
          )}
        </span>
      )}
    </button>
  );
}

export { classifyConfidence };