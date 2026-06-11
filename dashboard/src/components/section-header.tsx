import * as React from "react"

interface SectionHeaderProps {
  eyebrow?: string
  title: string
  description?: string
  actions?: React.ReactNode
  className?: string
}

export function SectionHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: SectionHeaderProps) {
  return (
    <header className={className}>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="space-y-1">
          {eyebrow && (
            <p className="text-[10px] font-medium tracking-[0.2em] text-muted-foreground uppercase">
              {eyebrow}
            </p>
          )}
          <h1 className="font-heading text-2xl font-medium tracking-tight md:text-3xl">
            {title}
          </h1>
          {description && (
            <p className="max-w-2xl text-sm text-muted-foreground">
              {description}
            </p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </header>
  )
}
