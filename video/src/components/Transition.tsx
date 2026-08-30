/** The between-scene wipe. Short, quiet, and the same every time. */

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { theme } from "../theme";

export const Transition: React.FC<{ label?: string }> = ({ label }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const t = frame / Math.max(durationInFrames, 1);
  const width = interpolate(t, [0, 0.5, 1], [0, 100, 0], { extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ backgroundColor: theme.colors.bg }}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            height: 3,
            width: `${width}%`,
            background: theme.colors.accent,
            borderRadius: 3,
          }}
        />
        {label ? (
          <div
            style={{
              marginTop: theme.space(3),
              fontFamily: theme.fonts.mono,
              fontSize: theme.size.tiny,
              letterSpacing: 3,
              textTransform: "uppercase",
              color: theme.colors.inkFaint,
              opacity: interpolate(t, [0.2, 0.5, 0.8], [0, 1, 0], { extrapolateRight: "clamp" }),
            }}
          >
            {label}
          </div>
        ) : null}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
