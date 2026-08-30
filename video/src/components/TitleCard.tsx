/** Episode opener: series mark, topic, episode number, runtime. */

import React from "react";
import { theme } from "../theme";
import { Disclaimer, Stage, useEntrance, useSpringIn } from "./primitives";

export type TitleCardProps = {
  title: string;
  subtitle?: string;
  topicId?: string;
  episode?: string;
  paper?: string;
};

export const TitleCard: React.FC<TitleCardProps> = ({
  title,
  subtitle,
  topicId,
  episode,
  paper,
}) => {
  const hero = useSpringIn(4);
  const sub = useEntrance(16);
  const meta = useEntrance(26);
  return (
    <Stage>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
        <div
          style={{
            fontFamily: theme.fonts.mono,
            fontSize: theme.size.tiny,
            letterSpacing: 4,
            color: theme.colors.accent,
            marginBottom: theme.space(2),
          }}
        >
          AI SYSTEMS FORGE{topicId ? ` · ${topicId.toUpperCase()}` : ""}
          {episode ? ` · ${episode}` : ""}
        </div>
        <h1
          style={{
            ...hero,
            margin: 0,
            fontSize: theme.size.display,
            fontWeight: 700,
            letterSpacing: -1.5,
            lineHeight: 1.05,
            maxWidth: 980,
          }}
        >
          {title}
        </h1>
        {subtitle ? (
          <p
            style={{
              ...sub,
              fontSize: theme.size.body,
              color: theme.colors.inkMuted,
              marginTop: theme.space(3),
              maxWidth: 900,
              lineHeight: 1.4,
            }}
          >
            {subtitle}
          </p>
        ) : null}
        {paper ? (
          <div
            style={{
              ...meta,
              marginTop: theme.space(4),
              fontFamily: theme.fonts.mono,
              fontSize: theme.size.tiny,
              color: theme.colors.brass,
            }}
          >
            source · {paper}
          </div>
        ) : null}
      </div>
      <Disclaimer />
    </Stage>
  );
};
