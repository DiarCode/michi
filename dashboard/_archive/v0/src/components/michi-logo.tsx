import { cn } from "@/lib/utils"

interface MichiLogoProps {
  className?: string
  showWordmark?: boolean
}

export function MichiLogo({ className, showWordmark = true }: MichiLogoProps) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span
        aria-hidden
        className="relative grid size-7 place-items-center rounded-2xl bg-primary text-primary-foreground shadow-sm ring-1 ring-foreground/10"
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="size-4"
        >
          <path
            d="M4 18 C 7 8, 11 6, 14 8 L 20 12 L 14 16 C 11 18, 7 16, 4 18 Z"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle cx="17" cy="12" r="1.4" fill="currentColor" />
        </svg>
      </span>
      {showWordmark && (
        <span className="font-heading text-lg font-medium leading-none tracking-tight">
          Michi
        </span>
      )}
    </div>
  )
}
