/**
 * Burned-in captions, generated from the narration text and its Kokoro
 * timings. Accessibility, and the reason these episodes work on mute.
 *
 * Toggleable with the `captions` prop on `Episode` — the series ships them on.
 */

import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { theme } from "../theme";

export type CaptionCue = { start: number; end: number; text: string };

export const Captions: React.FC<{ cues: CaptionCue[] }> = ({ cues }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const cue = cues.find((c) => t >= c.start && t < c.end);
  if (!cue) return null;
  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: theme.space(4.5),
        display: "flex",
        justifyContent: "center",
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          maxWidth: 1000,
          textAlign: "center",
          background: "rgba(8, 14, 22, 0.86)",
          border: `1px solid ${theme.colors.line}`,
          borderRadius: 10,
          padding: `${theme.space(1.25)}px ${theme.space(2.5)}px`,
          fontFamily: theme.fonts.sans,
          fontSize: 23,
          lineHeight: 1.35,
          color: theme.colors.ink,
        }}
      >
        {cue.text}
      </div>
    </div>
  );
};
