import { Button as ButtonPrimitive } from "@base-ui/react/button"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "group/button inline-flex shrink-0 items-center justify-center rounded-full font-semibold whitespace-nowrap transition-all outline-none select-none focus-visible:ring-3 focus-visible:ring-michi-lime/50 active:not-aria-[haspopup]:translate-y-px disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default: "bg-michi-dark text-white hover:bg-michi-dark/90 shadow-sm",
        outline:
          "border-2 border-michi-dark bg-white text-michi-dark hover:bg-michi-warm",
        secondary:
          "bg-michi-warm text-michi-dark border border-michi-border hover:bg-michi-border",
        ghost:
          "text-michi-body hover:bg-michi-warm hover:text-michi-dark",
        destructive:
          "bg-michi-red/10 text-michi-red hover:bg-michi-red/20 border border-michi-red/30",
        lime:
          "bg-michi-lime text-michi-dark hover:bg-michi-lime-dark shadow-sm",
        link: "text-michi-dark underline-offset-4 hover:underline",
      },
      size: {
        default:
          "h-10 gap-2 px-5 py-2.5 text-sm has-data-[icon=inline-end]:pr-4 has-data-[icon=inline-start]:pl-4",
        xs: "h-7 gap-1 rounded-full px-3 text-xs has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-8 gap-1.5 rounded-full px-3.5 text-xs has-data-[icon=inline-end]:pr-2.5 has-data-[icon=inline-start]:pl-2.5 [&_svg:not([class*='size-'])]:size-3.5",
        lg: "h-11 gap-2 px-6 py-3 text-base has-data-[icon=inline-end]:pr-4 has-data-[icon=inline-start]:pl-4",
        icon: "size-10",
        "icon-xs":
          "size-7 rounded-full [&_svg:not([class*='size-'])]:size-3",
        "icon-sm":
          "size-8 rounded-full",
        "icon-lg": "size-11",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  ...props
}: ButtonPrimitive.Props & VariantProps<typeof buttonVariants>) {
  return (
    <ButtonPrimitive
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }