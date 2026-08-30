/** One idea, held on screen: a heading, a supporting line, optional bullets. */

import React from "react";
import { theme } from "../theme";
import { Disclaimer, Eyebrow, Rule, SceneTitle, Stage, useEntrance } from "./primitives";

export type ConceptSceneProps = {
  eyebrow?: string;
  title: string;
  body?: string;
  points?: string[];
  aside?: string;
};

export const ConceptScene: React.FC<ConceptSceneProps> = ({
  eyebrow,
  title,
  body,
  points = [],
  aside,
}) => {
  const bodyStyle = useEntrance(12);
  return (
    <Stage>
      {eyebrow ? <Eyebrow text={eyebrow} /> : null}
      <SceneTitle>{title}</SceneTitle>
      <Rule color={theme.colors.accentDim} />
      {body ? (
        <p
          style={{
            ...bodyStyle,
            fontSize: theme.size.body,
            lineHeight: 1.5,
            color: theme.colors.ink,
            maxWidth: 1020,
            margin: 0,
          }}
        >
          {body}
        </p>
      ) : null}
      <ul style={{ listStyle: "none", padding: 0, marginTop: theme.space(3) }}>
        {points.map((p, i) => (
          <Point key={p} text={p} index={i} />
        ))}
      </ul>
      {aside ? (
        <div
          style={{
            marginTop: "auto",
            fontFamily: theme.fonts.mono,
            fontSize: theme.size.tiny,
            color: theme.colors.inkFaint,
          }}
        >
          {aside}
        </div>
      ) : null}
      <Disclaimer />
    </Stage>
  );
};

const Point: React.FC<{ text: string; index: number }> = ({ text, index }) => {
  const style = useEntrance(20 + index * 12);
  return (
    <li
      style={{
        ...style,
        display: "flex",
        gap: theme.space(2),
        alignItems: "flex-start",
        fontSize: theme.size.small,
        lineHeight: 1.45,
        color: theme.colors.ink,
        marginBottom: theme.space(1.5),
        maxWidth: 1020,
      }}
    >
      <span style={{ color: theme.colors.accent, fontFamily: theme.fonts.mono }}>▸</span>
      <span>{text}</span>
    </li>
  );
};
