export function MichiLogo() {
  return (
    <div className="flex items-center gap-2 px-1 py-1">
      <div className="grid size-7 place-items-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
        <span className="font-heading text-sm font-medium">道</span>
      </div>
      <div className="flex flex-col leading-tight">
        <span className="font-heading text-sm font-medium">Michi</span>
        <span className="text-[10px] uppercase tracking-widest text-muted-foreground">Transit</span>
      </div>
    </div>
  )
}
