export function MichiLogo() {
  return (
    <div className="flex items-center gap-2 px-1 py-1">
      <div className="flex size-7 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
        <span className="font-heading text-sm font-bold">M</span>
      </div>
      <div className="flex flex-col leading-tight">
        <span className="font-heading text-sm font-semibold text-blue-600 dark:text-blue-400">
          Michi
        </span>
        <span className="text-[10px] tracking-widest text-muted-foreground uppercase">
          Transit
        </span>
      </div>
    </div>
  )
}