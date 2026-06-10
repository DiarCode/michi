import { cn } from "@/lib/utils"
import { HugeiconsIcon } from "@hugeicons/react"
import { Loading03Icon } from "@hugeicons/core-free-icons"

function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
  const { strokeWidth, ...rest } = props
  return (
    <HugeiconsIcon
      icon={Loading03Icon}
      strokeWidth={typeof strokeWidth === "number" ? strokeWidth : 2}
      role="status"
      aria-label="Loading"
      className={cn("size-4 animate-spin", className)}
      {...rest}
    />
  )
}

export { Spinner }
