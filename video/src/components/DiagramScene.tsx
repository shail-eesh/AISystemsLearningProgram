/**
 * Animated box-and-arrow diagrams: pipelines, layer descents, memory blocks.
 * Nodes appear in order and edges draw between them, so a mechanism is built
 * up on screen rather than presented finished.
 */

import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { theme } from "../theme";
import { Disclaimer, Eyebrow, SceneTitle, Stage } from "./primitives";

export type Node = {
  id: string;
  label: string;
  sub?: string;
  /** 0-1 fractions of the canvas. */
  x: number;
  y: number;
  w?: number;
  h?: number;
  color?: string;
  appearAt?: number;
};

export type Edge = {
  from: string;
  to: string;
  label?: string;
  appearAt?: number;
  dashed?: boolean;
};

export type DiagramSceneProps = {
  eyebrow?: string;
  title: string;
  nodes: Node[];
  edges?: Edge[];
  caption?: string;
};

const W = 1080;
const H = 420;

export const DiagramScene: React.FC<DiagramSceneProps> = ({
  eyebrow,
  title,
  nodes,
  edges = [],
  caption,
}) => {
  const frame = useCurrentFrame();
  const box = (n: Node) => {
    const w = (n.w ?? 0.2) * W;
    const h = (n.h ?? 0.16) * H;
    return { x: n.x * W - w / 2, y: n.y * H - h / 2, w, h, cx: n.x * W, cy: n.y * H };
  };
  const byId = Object.fromEntries(nodes.map((n) => [n.id, box(n)]));

  return (
    <Stage>
      {eyebrow ? <Eyebrow text={eyebrow} /> : null}
      <SceneTitle>{title}</SceneTitle>
      <div style={{ marginTop: theme.space(2), flex: 1 }}>
        <svg width={W} height={H}>
          <defs>
            <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3"
                    orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill={theme.colors.inkMuted} />
            </marker>
          </defs>
          {edges.map((e, i) => {
            const a = byId[e.from];
            const b = byId[e.to];
            if (!a || !b) return null;
            const at = e.appearAt ?? 30 + i * 10;
            const t = interpolate(frame - at, [0, 14], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            const horizontal = Math.abs(b.cx - a.cx) > Math.abs(b.cy - a.cy);
            const x1 = horizontal ? (b.cx > a.cx ? a.x + a.w : a.x) : a.cx;
            const y1 = horizontal ? a.cy : b.cy > a.cy ? a.y + a.h : a.y;
            const x2 = horizontal ? (b.cx > a.cx ? b.x : b.x + b.w) : b.cx;
            const y2 = horizontal ? b.cy : b.cy > a.cy ? b.y : b.y + b.h;
            return (
              <g key={`${e.from}-${e.to}`} opacity={t}>
                <line
                  x1={x1}
                  y1={y1}
                  x2={x1 + (x2 - x1) * t}
                  y2={y1 + (y2 - y1) * t}
                  stroke={theme.colors.inkMuted}
                  strokeWidth={2}
                  strokeDasharray={e.dashed ? "6 5" : undefined}
                  markerEnd={t > 0.95 ? "url(#arrow)" : undefined}
                />
                {e.label ? (
                  <text
                    x={(x1 + x2) / 2}
                    y={(y1 + y2) / 2 - 8}
                    textAnchor="middle"
                    fill={theme.colors.inkFaint}
                    fontFamily={theme.fonts.mono}
                    fontSize={15}
                  >
                    {e.label}
                  </text>
                ) : null}
              </g>
            );
          })}
          {nodes.map((n, i) => {
            const b = byId[n.id];
            const at = n.appearAt ?? 8 + i * 10;
            const t = interpolate(frame - at, [0, 12], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            const colour = n.color ?? theme.colors.accent;
            return (
              <g key={n.id} opacity={t}>
                <rect
                  x={b.x}
                  y={b.y}
                  width={b.w}
                  height={b.h}
                  rx={12}
                  fill={theme.colors.bgPanel}
                  stroke={colour}
                  strokeWidth={2}
                />
                <text
                  x={b.cx}
                  y={b.cy + (n.sub ? -4 : 6)}
                  textAnchor="middle"
                  fill={theme.colors.ink}
                  fontFamily={theme.fonts.sans}
                  fontSize={20}
                  fontWeight={600}
                >
                  {n.label}
                </text>
                {n.sub ? (
                  <text
                    x={b.cx}
                    y={b.cy + 20}
                    textAnchor="middle"
                    fill={theme.colors.inkMuted}
                    fontFamily={theme.fonts.mono}
                    fontSize={15}
                  >
                    {n.sub}
                  </text>
                ) : null}
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
