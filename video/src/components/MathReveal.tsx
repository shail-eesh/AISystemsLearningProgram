/**
 * The three-way formula reveal (MASTER_PLAN §5.1): a symbol is never shown
 * before its meaning.
 *
 *   1. plain English sentence
 *   2. the symbolic equation
 *   3. the line of code that implements it
 *
 * Each stage fades in on its own beat, driven by `stageFrames` so the reveal
 * tracks the narration rather than a guessed timer.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { theme } from "../theme";
import { Disclaimer, Eyebrow, SceneTitle, Stage, useEntrance } from "./primitives";

export type MathRevealProps = {
  eyebrow?: string;
  title: string;
  english: string;
  equation: string;
  code: string;
  note?: string;
  /** Frame at which each of the three stages appears. */
  stageFrames?: [number, number, number];
};

const StageRow: React.FC<{
  label: string;
  children: React.ReactNode;
  appearAt: number;
  mono?: boolean;
  emphasis?: string;
  size?: number;
}> = ({ label, children, appearAt, mono, emphasis, size }) => {
  const frame = useCurrentFrame();
  const style = useEntrance(appearAt);
  const visible = frame >= appearAt;
  return (
    <div
      style={{
        ...style,
        opacity: visible ? style.opacity : 0,
        display: "grid",
        gridTemplateColumns: "116px 1fr",
        gap: theme.space(3),
        alignItems: "baseline",
        marginBottom: theme.space(3),
      }}
    >
      <div
        style={{
          fontFamily: theme.fonts.mono,
          fontSize: theme.size.tiny,
          letterSpacing: 2,
          textTransform: "uppercase",
          color: theme.colors.inkFaint,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: mono ? theme.fonts.mono : theme.fonts.sans,
          fontSize: size ?? theme.size.small,
          lineHeight: 1.45,
          color: emphasis ?? theme.colors.ink,
        }}
      >
        {children}
      </div>
    </div>
  );
};

export const MathReveal: React.FC<MathRevealProps> = ({
  eyebrow,
  title,
  english,
  equation,
  code,
  note,
  stageFrames = [10, 70, 140],
}) => (
  <Stage>
    {eyebrow ? <Eyebrow text={eyebrow} /> : null}
    <SceneTitle>{title}</SceneTitle>
    <div style={{ marginTop: theme.space(4) }}>
      <StageRow label="in words" appearAt={stageFrames[0]}>
        {english}
      </StageRow>
      <StageRow
        label="in symbols"
        appearAt={stageFrames[1]}
        mono
        emphasis={theme.colors.brass}
        size={theme.size.heading}
      >
        {equation}
      </StageRow>
      <StageRow label="in code" appearAt={stageFrames[2]} mono emphasis={theme.colors.accent}>
        <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{code}</pre>
      </StageRow>
    </div>
    {note ? (
      <div
        style={{
          marginTop: "auto",
          fontSize: theme.size.tiny,
          color: theme.colors.inkMuted,
          fontFamily: theme.fonts.mono,
        }}
      >
        {note}
      </div>
    ) : null}
    <Disclaimer />
  </Stage>
);
