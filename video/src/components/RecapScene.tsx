/** Closing bookend: three bullets and "what breaks if we skip this". */

import React from "react";
import { theme } from "../theme";
import { Disclaimer, Eyebrow, Rule, SceneTitle, Stage, useEntrance } from "./primitives";

export type RecapSceneProps = {
  eyebrow?: string;
  title?: string;
  points: string[];
  ifSkipped: string;
  next?: string;
};

/** 720p is 720p: long recaps have to shrink rather than run off the bottom. */
const fit = (points: string[], ifSkipped: string) => {
  const load = points.join(" ").length + ifSkipped.length * 0.6;
  if (load > 520) return { point: 22, skip: 19, gap: 1.1 };
  if (load > 380) return { point: 25, skip: 21, gap: 1.4 };
  return { point: theme.size.body, skip: theme.size.small, gap: 2 };
};

export const RecapScene: React.FC<RecapSceneProps> = ({
  eyebrow,
  title = "Recap",
  points,
  ifSkipped,
  next,
}) => {
  const skip = useEntrance(20 + points.length * 14);
  const nextStyle = useEntrance(32 + points.length * 14);
  const sizes = fit(points, ifSkipped);
  return (
    <Stage>
      {eyebrow ? <Eyebrow text={eyebrow} /> : null}
      <SceneTitle>{title}</SceneTitle>
      <Rule color={theme.colors.accentDim} />
      <ol style={{ padding: 0, listStyle: "none", margin: 0 }}>
        {points.map((p, i) => (
          <RecapPoint key={p} text={p} index={i} size={sizes.point} gap={sizes.gap} />
        ))}
      </ol>
      <div
        style={{
          ...skip,
          marginTop: theme.space(2.5),
          background: theme.colors.brassDim,
          border: `1px solid ${theme.colors.brass}`,
          borderRadius: theme.radius,
          padding: theme.space(2.25),
        }}
      >
        <div
          style={{
            fontFamily: theme.fonts.mono,
            fontSize: theme.size.tiny,
            letterSpacing: 2,
            textTransform: "uppercase",
            color: theme.colors.brass,
            marginBottom: theme.space(1),
          }}
        >
          what breaks if we skip this
        </div>
        <div style={{ fontSize: sizes.skip, lineHeight: 1.4 }}>{ifSkipped}</div>
      </div>
      {next ? (
        <div
          style={{
            ...nextStyle,
            // Top-right, not bottom: the bottom band belongs to the captions.
            position: "absolute",
            top: theme.space(9),
            right: theme.space(9),
            fontFamily: theme.fonts.mono,
            fontSize: theme.size.tiny,
            color: theme.colors.accent,
            textAlign: "right",
            maxWidth: 460,
          }}
        >
          next · {next}
        </div>
      ) : null}
      <Disclaimer />
    </Stage>
  );
};

const RecapPoint: React.FC<{ text: string; index: number; size: number; gap: number }> = ({
  text,
  index,
  size,
  gap,
}) => {
  const style = useEntrance(10 + index * 14);
  return (
    <li
      style={{
        ...style,
        display: "flex",
        gap: theme.space(2.5),
        alignItems: "flex-start",
        fontSize: size,
        lineHeight: 1.35,
        marginBottom: theme.space(gap),
        maxWidth: 1080,
      }}
    >
      <span
        style={{
          fontFamily: theme.fonts.mono,
          color: theme.colors.accent,
          fontSize: size,
          minWidth: 30,
        }}
      >
        {index + 1}.
      </span>
      <span>{text}</span>
    </li>
  );
};
