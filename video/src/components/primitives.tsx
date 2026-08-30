/**
 * Shared building blocks: the animation helpers and layout chrome every
 * scene uses. Keeping them here is what stops each episode from re-inventing
 * its own easing and padding.
 */

import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { theme } from "../theme";

/** Fade + rise, the series' single entrance animation. */
export const useEntrance = (delayFrames = 0, durationFrames = 18) => {
  const frame = useCurrentFrame();
  const t = interpolate(frame - delayFrames, [0, durationFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return { opacity: t, transform: `translateY(${(1 - t) * 14}px)` };
};

/** A gentler, springier entrance for hero elements. */
export const useSpringIn = (delayFrames = 0) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - delayFrames, fps, config: { damping: 200 } });
  return { opacity: s, transform: `scale(${0.96 + 0.04 * s})` };
};

export const Stage: React.FC<{
  children: React.ReactNode;
  padded?: boolean;
  background?: string;
}> = ({ children, padded = true, background }) => (
  <AbsoluteFill
    style={{
      backgroundColor: background ?? theme.colors.bg,
      color: theme.colors.ink,
      fontFamily: theme.fonts.sans,
      padding: padded ? theme.space(9) : 0,
      // Leave the bottom band clear for burned-in captions (see Captions.tsx).
      paddingBottom: padded ? theme.space(15) : 0,
      display: "flex",
      flexDirection: "column",
    }}
  >
    {children}
  </AbsoluteFill>
);

/** The small top-left label that tells you where you are in the course. */
export const Eyebrow: React.FC<{ text: string }> = ({ text }) => (
  <div
    style={{
      fontFamily: theme.fonts.mono,
      fontSize: theme.size.tiny,
      letterSpacing: 2.5,
      textTransform: "uppercase",
      color: theme.colors.accent,
      marginBottom: theme.space(2),
    }}
  >
    {text}
  </div>
);

export const SceneTitle: React.FC<{ children: React.ReactNode; delay?: number }> = ({
  children,
  delay = 0,
}) => {
  const style = useEntrance(delay);
  return (
    <h2
      style={{
        ...style,
        margin: 0,
        fontSize: theme.size.heading,
        fontWeight: 650,
        lineHeight: 1.15,
        letterSpacing: -0.5,
      }}
    >
      {children}
    </h2>
  );
};

export const Rule: React.FC<{ color?: string }> = ({ color }) => (
  <div
    style={{
      height: 2,
      background: color ?? theme.colors.line,
      margin: `${theme.space(3)}px 0`,
      borderRadius: 2,
    }}
  />
);

export const Panel: React.FC<{
  children: React.ReactNode;
  accent?: string;
  style?: React.CSSProperties;
}> = ({ children, accent, style }) => (
  <div
    style={{
      background: theme.colors.bgPanel,
      border: `1px solid ${accent ?? theme.colors.line}`,
      borderRadius: theme.radius,
      padding: theme.space(3.5),
      ...style,
    }}
  >
    {children}
  </div>
);

/** The persistent footer disclaimer — AlphaDesk is never presented as real. */
export const Disclaimer: React.FC = () => (
  <div
    style={{
      position: "absolute",
      bottom: theme.space(2),
      right: theme.space(4),
      fontSize: 13,
      color: theme.colors.inkFaint,
      fontFamily: theme.fonts.mono,
    }}
  >
    AlphaDesk · fictional educational simulation
  </div>
);
