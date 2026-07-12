/** Tailwind for the TARA site — utilities only.
 *
 * - preflight OFF: Tailwind's CSS reset is disabled so the existing hand-written
 *   styles stay pixel-identical. Utilities are purely additive.
 * - colors map to the site's CSS variables, so `bg-card`, `text-ink`, `text-teal`
 *   etc. automatically follow the light/dark theme toggle.
 * - safelist: a starter kit of common utilities compiled even before first use,
 *   so new UI work can use them immediately without a rebuild.
 *
 * Rebuild after adding new classes in the HTML:  ./build-tw.sh
 */
module.exports = {
  content: ["./*.html"],
  corePlugins: { preflight: false },
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)", card: "var(--card)", line: "var(--line)", line2: "var(--line2)",
        ink: "var(--ink)", mut: "var(--mut)", faint: "var(--faint)",
        indigo: "var(--indigo)", teal: "var(--teal)", amber: "var(--amber)",
        red: "var(--red)", violet: "var(--violet)", green: "var(--green)",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
    },
  },
  safelist: [
    "flex", "flex-col", "items-center", "justify-between", "justify-center",
    "gap-1", "gap-2", "gap-3", "gap-4", "hidden", "block", "w-full",
    "p-2", "p-3", "p-4", "px-2", "px-3", "px-4", "py-1", "py-2", "py-3",
    "mt-1", "mt-2", "mt-3", "mt-4", "mb-2", "mb-4", "ml-auto",
    "rounded-md", "rounded-lg", "rounded-xl", "rounded-full", "border",
    "text-xs", "text-sm", "text-base", "text-lg", "font-medium", "font-semibold", "font-bold",
    "font-mono", "text-center", "truncate", "cursor-pointer", "select-none",
    "bg-bg", "bg-card", "text-ink", "text-mut", "text-faint",
    "text-indigo", "text-teal", "text-amber", "text-red", "text-violet",
    "border-line", "border-line2",
  ],
};
