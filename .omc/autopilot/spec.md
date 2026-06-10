# Michi Dashboard — Production-Ready Polish Spec

## Context

The Michi dashboard (a React 19 + Vite + Tailwind 3 + shadcn-style + base-ui app
backed by a FastAPI transit-intelligence service) was partially redesigned but
the result still looks raw: a top bar with pill nav, KPI cards, an oversized
brand block, ad-hoc colored chips, and a Command Center page that stacks
unevenly-sized cards. The user wants the whole UI taken to a production-ready
bar — coherent design system, shadcn components throughout, real spacing and
typography, polished empty/loading/error states, and a working dark/light
theme.

The user invoked `/autopilot` and the design choice they typed
(`npx shadcn@latest init --preset b6GfNnvwA`) points to a non-public preset ID.
Public registry confirms the item does not exist
(`https://ui.shadcn.com/r/styles/new-york-v4/b6GfNnvwA.json` returns 404).
The user confirmed falling back to "iterate within current setup" once that
became clear. The dashboard already imports `shadcn/tailwind.css`, uses
oklch design tokens, base-ui primitives, and the shadcn components.json
`base-maia` style — so a full re-init would risk data loss. We polish what is
there instead of ripping it out.

## Goals (production-ready)

1. **Coherent design system** — one set of oklch tokens already exists; lean
   on them. Set `radius` back to a uniform value, add `border`/`input`/`ring`
   derived shades, and use `bg-card` / `bg-muted` / `bg-accent` consistently
   instead of hard-coded `bg-chart-*` for chrome.
2. **Real shadcn components** — install (via the existing shadcn CLI) every
   primitive the app needs: button, card, badge, dialog, tabs, sheet, table,
   separator, skeleton, tooltip, popover, select, dropdown-menu, switch,
   sonner, progress, scroll-area, sidebar, breadcrumb, command, input, label,
   field, item, kbd, spinner, empty, chart. Pull them into the same
   `src/components/ui/` folder using `npx shadcn@latest add <name>` with the
   `-y` flag.
3. **App shell rebuild** — sidebar layout (shadcn `sidebar` primitive) with
   a top bar containing the role switcher, theme switcher, connection
   indicator, and a global Command-K launcher. Replace the giant "M" logo
   block with the real `Michi` wordmark + tagline.
4. **Every page** — Command Center, Live Map, Alerts, Simulation, Forecast,
   Executive, Settings — gets rewritten against shadcn primitives with real
   empty states, loading skeletons, and error boundaries.
5. **Polished typography & spacing** — single font stack (Geist Variable
   already installed; drop IBM Plex Sans to declutter), consistent 4/6/8 px
   rhythm, `text-balance` on headings, antialiased rendering.
6. **Working theme** — light & dark both look like a real product, not a
   flashlight, using a tested color combination (e.g. neutral/zinc base with
   a teal-cyan primary for the transit feel).
7. **Verify with Playwright** — start vite dev server, take full-page
   screenshots of every route in light + dark, open them, look for layout
   breaks, fix, repeat until clean.

## Non-Goals

- No backend changes — FastAPI surface stays as-is.
- No new features — the only changes are visual/UX polish and component
  composition.
- No re-architected state management — zustand stores stay.

## Acceptance Criteria

- [ ] `npm run build` succeeds with zero TS errors and zero warnings.
- [ ] `npm run test` passes (vitest).
- [ ] Every route (`/`, `/map`, `/alerts`, `/simulation`, `/forecast`,
  `/executive`, `/settings`) renders without console errors in both light
  and dark themes.
- [ ] Playwright screenshots of each route look like a real product: no
  overlapping text, no default browser scrollbars visible, no `M` placeholder
  logos, no giant emoji (🔴 ⚡) in headings, consistent card padding, and a
  visible brand mark.
- [ ] Toggling role and theme persists across reload (localStorage).
- [ ] No leftover `bg-chart-2` used for primary buttons; all primary CTAs
  use `bg-primary`.

## Implementation Plan

### Step 1 — Theme & foundation (tokens, fonts)
- `npx shadcn@latest add --yes` for the full component list (button, card,
  badge, dialog, tabs, sheet, table, separator, skeleton, tooltip, popover,
  select, dropdown-menu, switch, sonner, progress, scroll-area, sidebar,
  breadcrumb, command, input, label, field, item, kbd, spinner, empty, chart,
  alert, accordion, avatar, button-group, card, hover-card, navigation-menu,
  pagination, radio-group, resizable, slider, toggle, toggle-group, textarea).
  Let the shadcn CLI install to `src/components/ui/`.
- Update `src/styles/globals.css` to use a single, well-considered palette:
  - Light: warm white background (`oklch(0.99 0 0)`), zinc card, indigo
    primary, neutral foreground.
  - Dark: deep slate background, slightly lifted card, indigo primary.
  - Chart colors: a true 5-step palette (cyan/teal/violet/amber/rose).
- Add a `border` token explicitly (currently missing in dark).
- Remove the `IBM Plex Sans` font import; keep Geist Variable.
- Set `--radius: 0.75rem` and use it everywhere.

### Step 2 — App shell
- Replace the current fixed top bar with shadcn `sidebar` (collapsible) plus a
  slim `topbar` that holds connection, theme, role, and `command-k`.
- Add `<kbd>⌘K</kbd>` launcher backed by shadcn `command`.
- Logo: `Michi` wordmark in semibold with a small icon (the `M` square, but
  properly sized, with a transit glyph inside).
- Theme switcher: `sun`/`moon`/`system` segmented control.

### Step 3 — Command Center
- Hero header with role-aware greeting + last-updated timestamp.
- KPI grid (6 cards) using `card` with a header (icon + label) and a large
  metric.
- Section: "Live Operations" — 2-col layout: PredictiveHeatmap on the left,
  AnomalyPulse + DriftMonitor + ConnectionProtectionPanel stacked on the
  right.
- Section: "Active Playbooks" — table of open playbooks using shadcn `table`.
- Section: "Optimization Suggestions" — accordion with action buttons.
- Section: "Simulation Engine" — live KPI strip.
- Loading states: skeleton grids; error states: alert with retry.

### Step 4 — Live Map
- Full-bleed MapLibre map (already there) with floating top-left
  `card` containing map controls (toggle layers, zoom, search) and a
  bottom drawer (shadcn `sheet`) that opens station details.

### Step 5 — Alerts
- Tabbed view: Active / Resolved / All.
- Each alert row is a `card` with priority badge, route badge, timestamp, and
  inline playbook actions.
- Top bar filter: severity, route, district, time window.

### Step 6 — Simulation
- Page header + start/stop/playback controls.
- Live metric strip: tick, MAPE, drift, queue.
- Charts in tabs (Forecast / Drift / Anomalies).
- "What-if" panel: sliders (bus frequency, headway) that drive a real
  scenario run.

### Step 7 — Forecast
- Date-range picker (shadcn `calendar` + `popover`), route selector, horizon
  selector.
- Forecast chart with confidence band.
- Per-route table with predicted vs. actual.

### Step 8 — Executive
- KPI banner (3-4 top-line metrics).
- Charts: Ridership trend, On-time trend, Cost per ride, Net promoter (synth
  numbers).
- Footnote card: methodology + caveats.

### Step 9 — Settings
- Profile section (read-only), Theme, Notifications, Data refresh, About.

### Step 10 — Verify
- Start dev server in background.
- Playwright: open each route in light + dark, full-page screenshot, save
  into `.playwright-mcp/`.
- Open the screenshots, find problems, fix, repeat.

## Risks

- `npx shadcn@latest add` may overwrite existing `card.tsx`/`button.tsx`/
  `badge.tsx` already in `src/components/ui/`. Mitigation: back up the
  existing files before adding, then merge semantically.
- The map components use raw `maplibre-gl`; the redesign must not break the
  map's event bindings.
- Recharts tooltip override must keep working in both themes.
