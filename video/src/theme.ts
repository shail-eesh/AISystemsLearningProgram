/**
 * The one place the series' look is defined.
 *
 * A "terminal lecture" aesthetic: near-black slate ground, a restrained cyan
 * accent for structure, brass for emphasis, and exactly two type families.
 * Every component reads from here, which is why ~170 episodes generated over
 * three weeks by different sessions still look like one series.
 *
 * Colours are chosen for contrast on a projector and for the colour-blind:
 * the accent/brass pair is distinguishable in deuteranopia, and nothing
 * encodes meaning by hue alone.
 */

export const theme = {
  colors: {
    bg: "#0d1520",
    bgPanel: "#141f2c",
    bgPanelAlt: "#1a2634",
    ink: "#e8eef5",
    inkMuted: "#93a3b4",
    inkFaint: "#5c6b7c",
    accent: "#4ec9d6",
    accentDim: "#1f4c56",
    brass: "#e0b04e",
    brassDim: "#4a3c18",
    good: "#5ac48c",
    warn: "#e08a4e",
    bad: "#e05e6b",
    grid: "#22303f",
    line: "#2c3b4c",
  },
  fonts: {
    sans: '"Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", sans-serif',
    mono: '"JetBrains Mono", "SF Mono", "Cascadia Mono", "Menlo", "Consolas", monospace',
  },
  size: {
    display: 68,
    title: 52,
    heading: 40,
    body: 30,
    small: 24,
    tiny: 19,
    code: 26,
  },
  space: (n: number) => n * 8,
  radius: 16,
  video: {
    width: 1280,
    height: 720,
    fps: 30,
  },
} as const;

export type Theme = typeof theme;

/** Frames a duration in seconds occupies at the series frame rate. */
export const seconds = (s: number): number => Math.round(s * theme.video.fps);

/** The series' minimum scene length — MASTER_PLAN §5.1: no scene under 8s. */
export const MIN_SCENE_SECONDS = 8;
