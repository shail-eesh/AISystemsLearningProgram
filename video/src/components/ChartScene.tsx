/**
 * Plots drawn as inline SVG — loss curves, benchmark bars, recall/latency.
 * No charting library: 170 episodes of one dependency version is a liability,
 * and these shapes are twenty lines each.
 */

import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { theme } from "../theme";
import { Disclaimer, Eyebrow, SceneTitle, Stage } from "./primitives";

export type Series = { label: string; values: number[]; color?: string };
export type Bar = { label: string; value: number; color?: string; note?: string };

export type ChartSceneProps = {
  eyebrow?: string;
  title: string;
  kind: "line" | "bar";
  series?: Series[];
  bars?: Bar[];
  yLabel?: string;
  xLabel?: string;
  /** Draw a horizontal reference line (a baseline, a tolerance). */
  reference?: { value: number; label: string };
  caption?: string;
  logScale?: boolean;
};

const W = 1080;
const H = 400;
const PAD = { l: 84, r: 28, t: 20, b: 52 };

export const ChartScene: React.FC<ChartSceneProps> = (props) => (
  <Stage>
    {props.eyebrow ? <Eyebrow text={props.eyebrow} /> : null}
    <SceneTitle>{props.title}</SceneTitle>
    <div style={{ marginTop: theme.space(2), flex: 1 }}>
      <svg width={W} height={H} role="img">
        {props.kind === "line" ? <LineChart {...props} /> : <BarChart {...props} />}
      </svg>
    </div>
    {props.caption ? (
      <div style={{ fontSize: theme.size.small, color: theme.colors.inkMuted }}>
        {props.caption}
      </div>
    ) : null}
    <Disclaimer />
  </Stage>
);

/** Bars in this series span 1e-7 to 1e2, so the format has to follow the value. */
const formatValue = (v: number): string => {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  if (a >= 0.01) return v.toFixed(2);
  return v.toExponential(1);
};

const axisText = {
  fill: theme.colors.inkFaint,
  fontFamily: theme.fonts.mono,
  fontSize: 15,
};

const LineChart: React.FC<ChartSceneProps> = ({ series = [], yLabel, xLabel, reference }) => {
  const frame = useCurrentFrame();
  const all = series.flatMap((s) => s.values).concat(reference ? [reference.value] : []);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const n = Math.max(...series.map((s) => s.values.length), 2);
  const x = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => H - PAD.b - ((v - lo) / span) * (H - PAD.t - PAD.b);
  const progress = interpolate(frame, [10, 100], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <g>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <line
            x1={PAD.l}
            x2={W - PAD.r}
            y1={y(lo + f * span)}
            y2={y(lo + f * span)}
            stroke={theme.colors.grid}
          />
          <text x={PAD.l - 12} y={y(lo + f * span) + 5} textAnchor="end" {...axisText}>
            {(lo + f * span).toFixed(2)}
          </text>
        </g>
      ))}
      {reference ? (
        <g>
          <line
            x1={PAD.l}
            x2={W - PAD.r}
            y1={y(reference.value)}
            y2={y(reference.value)}
            stroke={theme.colors.brass}
            strokeDasharray="7 6"
          />
          <text x={W - PAD.r} y={y(reference.value) - 10} textAnchor="end" {...axisText} fill={theme.colors.brass}>
            {reference.label}
          </text>
        </g>
      ) : null}
      {series.map((s, si) => {
        const upto = Math.max(2, Math.round(s.values.length * progress));
        const d = s.values
          .slice(0, upto)
          .map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(v)}`)
          .join(" ");
        const colour = s.color ?? (si === 0 ? theme.colors.accent : theme.colors.warn);
        return (
          <g key={s.label}>
            <path d={d} fill="none" stroke={colour} strokeWidth={3} strokeLinejoin="round" />
            <text x={PAD.l + 8} y={PAD.t + 20 + si * 24} fill={colour} fontFamily={theme.fonts.mono} fontSize={17}>
              — {s.label}
            </text>
          </g>
        );
      })}
      {yLabel ? (
        <text x={18} y={H / 2} transform={`rotate(-90 18 ${H / 2})`} textAnchor="middle" {...axisText}>
          {yLabel}
        </text>
      ) : null}
      {xLabel ? (
        <text x={(W + PAD.l) / 2} y={H - 12} textAnchor="middle" {...axisText}>
          {xLabel}
        </text>
      ) : null}
    </g>
  );
};

const BarChart: React.FC<ChartSceneProps> = ({ bars = [], logScale, reference }) => {
  const frame = useCurrentFrame();
  const grow = interpolate(frame, [8, 46], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const scale = (v: number) => (logScale ? Math.log10(Math.max(v, 1e-12)) : v);
  const values = bars.map((b) => scale(b.value));
  const rawHi = Math.max(...values, reference ? scale(reference.value) : -Infinity);
  // Log scales run negative for sub-unit values, so headroom is additive, never
  // multiplicative — `hi * 1.08` moves a negative maximum the wrong way.
  const lo = logScale ? Math.min(...values) - 0.5 : 0;
  const span = Math.max(rawHi - lo, 1e-9) * 1.14;
  const hi = lo + span;
  const bw = (W - PAD.l - PAD.r) / (bars.length * 1.6);
  const x = (i: number) => PAD.l + i * bw * 1.6 + bw * 0.3;
  const h = (v: number) => ((scale(v) - lo) / span) * (H - PAD.t - PAD.b);

  return (
    <g>
      <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} stroke={theme.colors.line} />
      {bars.map((b, i) => {
        const height = h(b.value) * grow;
        return (
          <g key={b.label}>
            <rect
              x={x(i)}
              y={H - PAD.b - height}
              width={bw}
              height={Math.max(height, 0)}
              fill={b.color ?? theme.colors.accent}
              rx={5}
            />
            <text
              x={x(i) + bw / 2}
              y={Math.max(PAD.t + 14, H - PAD.b - height - 12)}
              textAnchor="middle"
              {...axisText}
              fill={theme.colors.ink}
            >
              {formatValue(b.value)}
            </text>
            <text x={x(i) + bw / 2} y={H - PAD.b + 24} textAnchor="middle" {...axisText}>
              {b.label}
            </text>
            {b.note ? (
              <text x={x(i) + bw / 2} y={H - PAD.b + 44} textAnchor="middle" {...axisText} fill={theme.colors.inkFaint}>
                {b.note}
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
};
