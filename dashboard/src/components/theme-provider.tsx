/* eslint-disable react-refresh/only-export-components */
import * as React from "react"

type Theme = "dark" | "light" | "system"
type ResolvedTheme = "dark" | "light"

type ThemeProviderProps = {
  children: React.ReactNode
  defaultTheme?: Theme
  storageKey?: string
  disableTransitionOnChange?: boolean
  /**
   * Lock the app to light mode. When true, the provider will:
   *   - ignore stored `dark` / `system` preferences and resolve to "light"
   *   - never apply the `.dark` class to the root element
   *   - disable the ⌘D keyboard shortcut that toggles themes
   *   - tell the browser to advertise only the light color scheme
   *     (so native form controls, scrollbars, and auto-filled inputs
   *     never invert).
   */
  disableDarkMode?: boolean
}

type ThemeProviderState = {
  theme: Theme
  setTheme: (theme: Theme) => void
}

const COLOR_SCHEME_QUERY = "(prefers-color-scheme: dark)"
const THEME_VALUES: Theme[] = ["dark", "light", "system"]

const ThemeProviderContext = React.createContext<
  ThemeProviderState | undefined
>(undefined)

function isTheme(value: string | null): value is Theme {
  if (value === null) {
    return false
  }

  return THEME_VALUES.includes(value as Theme)
}

function getSystemTheme(): ResolvedTheme {
  if (window.matchMedia(COLOR_SCHEME_QUERY).matches) {
    return "dark"
  }

  return "light"
}

function disableTransitionsTemporarily() {
  const style = document.createElement("style")
  style.appendChild(
    document.createTextNode(
      "*,*::before,*::after{-webkit-transition:none!important;transition:none!important}"
    )
  )
  document.head.appendChild(style)

  return () => {
    document.getElementById(style.id)?.remove()
    window.getComputedStyle(document.body)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        style.remove()
      })
    })
  }
}

/** Strip any stored `dark` / `system` preference and force `light`. */
function readStoredTheme(storageKey: string, disableDarkMode: boolean): Theme {
  if (disableDarkMode) {
    // Clear any prior dark preference so reloads don't go dark again.
    try {
      localStorage.removeItem(storageKey)
    } catch {
      /* ignore */
    }
    return "light"
  }
  const stored = localStorage.getItem(storageKey)
  return isTheme(stored) ? stored : "light"
}

/** Tell the browser to advertise only the light color scheme. */
function syncColorSchemeMeta(disableDarkMode: boolean) {
  if (typeof document === "undefined") return
  let meta = document.querySelector<HTMLMetaElement>('meta[name="color-scheme"]')
  if (!meta) {
    meta = document.createElement("meta")
    meta.name = "color-scheme"
    document.head.appendChild(meta)
  }
  meta.content = disableDarkMode ? "only light" : "light dark"
}

export function ThemeProvider({
  children,
  defaultTheme = "light",
  storageKey = "theme",
  disableTransitionOnChange = true,
  disableDarkMode = false,
  ...props
}: ThemeProviderProps) {
  const [theme, setThemeState] = React.useState<Theme>(() =>
    readStoredTheme(storageKey, disableDarkMode)
  )

  const setTheme = React.useCallback(
    (nextTheme: Theme) => {
      // When dark mode is disabled, silently coerce to "light".
      const resolved = disableDarkMode && nextTheme !== "light" ? "light" : nextTheme
      try {
        localStorage.setItem(storageKey, resolved)
      } catch {
        /* ignore */
      }
      setThemeState(resolved)
    },
    [storageKey, disableDarkMode]
  )

  const applyTheme = React.useCallback(
    (nextTheme: Theme) => {
      const root = document.documentElement
      // When dark mode is disabled we always resolve to "light",
      // regardless of what nextTheme is.
      const resolvedTheme: ResolvedTheme = disableDarkMode
        ? "light"
        : nextTheme === "system"
          ? getSystemTheme()
          : nextTheme
      const restoreTransitions = disableTransitionOnChange
        ? disableTransitionsTemporarily()
        : null

      root.classList.remove("light", "dark")
      root.classList.add(resolvedTheme)

      if (restoreTransitions) {
        restoreTransitions()
      }
    },
    [disableTransitionOnChange, disableDarkMode]
  )

  React.useEffect(() => {
    applyTheme(theme)
    syncColorSchemeMeta(disableDarkMode)

    if (theme !== "system" || disableDarkMode) {
      return undefined
    }

    const mediaQuery = window.matchMedia(COLOR_SCHEME_QUERY)
    const handleChange = () => {
      applyTheme("system")
    }

    mediaQuery.addEventListener("change", handleChange)

    return () => {
      mediaQuery.removeEventListener("change", handleChange)
    }
  }, [theme, applyTheme, disableDarkMode])

  // ⌘D keyboard shortcut for toggling theme — disabled when dark mode is
  // disabled so the user can't accidentally flip into a broken state.
  React.useEffect(() => {
    if (disableDarkMode) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return
      if (event.metaKey || event.ctrlKey || event.altKey) return

      const target = event.target
      if (target instanceof HTMLElement) {
        if (target.isContentEditable) return
        if (target.closest("input, textarea, select, [contenteditable='true']")) return
      }

      if (event.key.toLowerCase() !== "d") return

      setThemeState((currentTheme) => {
        const nextTheme =
          currentTheme === "dark"
            ? "light"
            : currentTheme === "light"
              ? "dark"
              : getSystemTheme() === "dark"
                ? "light"
                : "dark"

        try {
          localStorage.setItem(storageKey, nextTheme)
        } catch {
          /* ignore */
        }
        return nextTheme
      })
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [storageKey, disableDarkMode])

  React.useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      if (event.storageArea !== localStorage) return
      if (event.key !== storageKey) return

      if (disableDarkMode) {
        // Re-stamp light to clear any dark flip from another tab.
        try {
          localStorage.setItem(storageKey, "light")
        } catch {
          /* ignore */
        }
        setThemeState("light")
        return
      }

      if (isTheme(event.newValue)) {
        setThemeState(event.newValue)
        return
      }
      setThemeState(defaultTheme)
    }

    window.addEventListener("storage", handleStorageChange)
    return () => window.removeEventListener("storage", handleStorageChange)
  }, [defaultTheme, storageKey, disableDarkMode])

  const value = React.useMemo(
    () => ({
      theme,
      setTheme,
    }),
    [theme, setTheme]
  )

  return (
    <ThemeProviderContext.Provider {...props} value={value}>
      {children}
    </ThemeProviderContext.Provider>
  )
}

export const useTheme = () => {
  const context = React.useContext(ThemeProviderContext)

  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider")
  }

  return context
}
