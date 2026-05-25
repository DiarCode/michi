import type { ReactNode, ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: "primary" | "outline" | "ghost";
  size?: "sm" | "md";
}

const variants = { primary: "bg-blue-600 text-white hover:bg-blue-700", outline: "border hover:bg-gray-50", ghost: "hover:bg-gray-100" };
const sizes = { sm: "px-3 py-1 text-sm", md: "px-4 py-2 text-sm" };

export function Button({ children, variant = "primary", size = "md", className = "", ...props }: Props) {
  return <button className={`rounded-lg font-medium transition ${variants[variant]} ${sizes[size]} ${className}`} {...props}>{children}</button>;
}
