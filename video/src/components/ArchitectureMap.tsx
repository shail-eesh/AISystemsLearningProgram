/**
 * The recap bookend: the AlphaDesk architecture with the current topic lit up.
 *
 * The map is defined once, here, and every episode passes only `highlight`.
 * That is what makes "here's where we are in the build" mean something across
 * 51 topics.
 */

import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { theme } from "../theme";
import { Disclaimer, Eyebrow, SceneTitle, Stage } from "./primitives";

type Block = { id: string; label: string; topics: string; row: number; col: number; span?: number };

/** MASTER_PLAN §6.2, laid out as a grid. */
export const ALPHADESK_BLOCKS: Block[] = [
  { id: "guardrails", label: "Guardrails perimeter", topics: "T28 · T44", row: 0, col: 0, span: 4 },
  { id: "router", label: "Semantic router", topics: "T36", row: 1, col: 0 },
  { id: "agent", label: "ReAct agent · CoT", topics: "T2 · T1", row: 1, col: 1 },
  { id: "tools", label: "Tool router", topics: "T24 · T40 · T17s", row: 1, col: 2 },
  { id: "orders", label: "Paper order book", topics: "P0.1 · T25", row: 1, col: 3 },
  { id: "retrieval", label: "RAG · GraphRAG", topics: "T6 · T20 · T37", row: 2, col: 0 },
  { id: "vectors", label: "Vector DB · driver", topics: "T5 · T50 · T43", row: 2, col: 1 },
  { id: "models", label: "AlphaSLM · FinTok", topics: "T15 · T4 · T30", row: 2, col: 2 },
  { id: "serving", label: "tickerd · gateway", topics: "T3 · T12 · T41", row: 2, col: 3 },
  { id: "data", label: "Loaders · features", topics: "P0.2 · T48 · T38", row: 3, col: 0 },
  { id: "training", label: "Training loop", topics: "P0.3 · T17 · T19", row: 3, col: 1 },
  { id: "eval", label: "Eval harness", topics: "T27 · T22", row: 3, col: 2 },
  { id: "multimodal", label: "Voice · vision", topics: "T33 · T34 · T32", row: 3, col: 3 },
];

export type ArchitectureMapProps = {
  eyebrow?: string;
  title?: string;
  /** Block ids to light up. */
  highlight: string[];
  caption?: string;
};

const W = 1080;
const H = 400;
const COLS = 4;

export const ArchitectureMap: React.FC<ArchitectureMapProps> = ({
  eyebrow,
  title = "Where we are in the build",
  highlight,
  caption,
}) => {
  const frame = useCurrentFrame();
  const rows = Math.max(...ALPHADESK_BLOCKS.map((b) => b.row)) + 1;
  const gap = 14;
  const cw = (W - gap * (COLS - 1)) / COLS;
  const ch = (H - gap * (rows - 1)) / rows;
  const pulse = 0.75 + 0.25 * Math.sin(frame / 7);

  return (
    <Stage>
      {eyebrow ? <Eyebrow text={eyebrow} /> : null}
      <SceneTitle>{title}</SceneTitle>
      <div style={{ marginTop: theme.space(2), flex: 1 }}>
        <svg width={W} height={H}>
          {ALPHADESK_BLOCKS.map((b, i) => {
            const on = highlight.includes(b.id);
            const w = cw * (b.span ?? 1) + gap * ((b.span ?? 1) - 1);
            const x = b.col * (cw + gap);
            const y = b.row * (ch + gap);
            const t = interpolate(frame - i * 3, [0, 10], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            return (
              <g key={b.id} opacity={t}>
                <rect
                  x={x}
                  y={y}
                  width={w}
                  height={ch}
                  rx={12}
                  fill={on ? theme.colors.accentDim : theme.colors.bgPanel}
                  stroke={on ? theme.colors.accent : theme.colors.line}
                  strokeWidth={on ? 3 : 1.5}
                  opacity={on ? pulse : 0.85}
                />
                <text
                  x={x + w / 2}
                  y={y + ch / 2 - 4}
                  textAnchor="middle"
                  fill={on ? theme.colors.ink : theme.colors.inkMuted}
                  fontFamily={theme.fonts.sans}
                  fontSize={19}
                  fontWeight={on ? 700 : 500}
                >
                  {b.label}
                </text>
                <text
                  x={x + w / 2}
                  y={y + ch / 2 + 20}
                  textAnchor="middle"
                  fill={on ? theme.colors.accent : theme.colors.inkFaint}
                  fontFamily={theme.fonts.mono}
                  fontSize={14}
                >
                  {b.topics}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      {caption ? (
        <div style={{ fontSize: theme.size.small, color: theme.colors.inkMuted }}>{caption}</div>
      ) : null}
      <Disclaimer />
    </Stage>
  );
};
