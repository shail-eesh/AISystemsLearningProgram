/**
 * Line-highlighted code. The active line is bright, the rest is dimmed, and
 * long files are revealed progressively rather than dumped.
 *
 * `highlights` is a list of [startFrame, lineNumbers] pairs, so the highlight
 * follows the narration exactly (the timings come from Kokoro, not guesses).
 */

import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { theme } from "../theme";
import { Disclaimer, Eyebrow, SceneTitle, Stage } from "./primitives";

export type Highlight = { at: number; lines: number[]; caption?: string };

export type CodeWalkthroughProps = {
  eyebrow?: string;
  title: string;
  filename?: string;
  code: string;
  highlights?: Highlight[];
  /** Reveal lines progressively: line n appears at revealStart + n*revealEvery. */
  revealStart?: number;
  revealEvery?: number;
  fontSize?: number;
};

export const CodeWalkthrough: React.FC<CodeWalkthroughProps> = ({
  eyebrow,
  title,
  filename,
  code,
  highlights = [],
  revealStart = 6,
  revealEvery = 2,
  fontSize,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const lines = code.replace(/\n+$/, "").split("\n");

  // Highlight `at` values are authored as relative weights, then stretched over
  // the scene's real length — which only exists once the narration is
  // synthesised. Without this, a walkthrough races ahead of the voice.
  const maxAt = Math.max(1, ...highlights.map((h) => h.at));
  const scaled = highlights.map((h) => ({
    ...h,
    at: Math.round((h.at / (maxAt * 1.12)) * durationInFrames),
  }));
  const active = scaled.filter((h) => frame >= h.at).slice(-1)[0];
  const activeLines = new Set(active?.lines ?? []);

  // Fit the block: monospace is ~0.605em wide, and lines are 1.45em tall.
  const longest = Math.max(...lines.map((l) => l.length), 1);
  const byWidth = Math.floor(1090 / (longest * 0.605));
  const byHeight = Math.floor(310 / (lines.length * 1.45));
  const size = fontSize ?? Math.max(14, Math.min(theme.size.code, byWidth, byHeight));

  return (
    <Stage>
      {eyebrow ? <Eyebrow text={eyebrow} /> : null}
      <SceneTitle>{title}</SceneTitle>
      <div
        style={{
          marginTop: theme.space(2.5),
          background: theme.colors.bgPanel,
          border: `1px solid ${theme.colors.line}`,
          borderRadius: theme.radius,
          overflow: "hidden",
          flex: 1,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {filename ? (
          <div
            style={{
              padding: `${theme.space(1.25)}px ${theme.space(2.5)}px`,
              borderBottom: `1px solid ${theme.colors.line}`,
              fontFamily: theme.fonts.mono,
              fontSize: theme.size.tiny,
              color: theme.colors.inkMuted,
              background: theme.colors.bgPanelAlt,
            }}
          >
            {filename}
          </div>
        ) : null}
        <div style={{ padding: theme.space(2), overflow: "hidden" }}>
          {lines.map((line, i) => {
            const n = i + 1;
            const revealed = frame >= revealStart + i * revealEvery;
            const isActive = activeLines.has(n);
            const dim = activeLines.size > 0 && !isActive;
            return (
              <div
                key={`${n}-${line}`}
                style={{
                  display: "flex",
                  gap: theme.space(2),
                  fontFamily: theme.fonts.mono,
                  fontSize: size,
                  lineHeight: 1.45,
                  opacity: revealed ? (dim ? 0.3 : 1) : 0,
                  background: isActive ? theme.colors.accentDim : "transparent",
                  borderLeft: `3px solid ${isActive ? theme.colors.accent : "transparent"}`,
                  paddingLeft: theme.space(1.5),
                  borderRadius: 4,
                  transition: "opacity 120ms linear",
                  whiteSpace: "pre",
                }}
              >
                <span style={{ color: theme.colors.inkFaint, width: 34, textAlign: "right" }}>
                  {n}
                </span>
                <span style={{ color: isActive ? theme.colors.ink : theme.colors.inkMuted }}>
                  {line || " "}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      {active?.caption ? (
        <div
          style={{
            marginTop: theme.space(2),
            fontSize: theme.size.small,
            color: theme.colors.brass,
            fontFamily: theme.fonts.sans,
          }}
        >
          {active.caption}
        </div>
      ) : null}
      <Disclaimer />
    </Stage>
  );
};
