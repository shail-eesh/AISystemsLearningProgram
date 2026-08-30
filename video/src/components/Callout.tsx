/** A gotcha, a warning, or a "this is the bug everyone hits" aside. */

import React from "react";
import { theme } from "../theme";
import { Disclaimer, Stage, useSpringIn } from "./primitives";

export type CalloutProps = {
  kind?: "gotcha" | "warning" | "insight";
  heading: string;
  body: string;
  code?: string;
};

const KINDS = {
  gotcha: { colour: theme.colors.warn, label: "gotcha" },
  warning: { colour: theme.colors.bad, label: "warning" },
  insight: { colour: theme.colors.good, label: "insight" },
} as const;

export const Callout: React.FC<CalloutProps> = ({ kind = "gotcha", heading, body, code }) => {
  const style = useSpringIn(3);
  const { colour, label } = KINDS[kind];
  return (
    <Stage>
      <div style={{ flex: 1, display: "flex", alignItems: "center" }}>
        <div
          style={{
            ...style,
            width: "100%",
            background: theme.colors.bgPanel,
            border: `2px solid ${colour}`,
            borderRadius: theme.radius,
            padding: theme.space(5),
          }}
        >
          <div
            style={{
              fontFamily: theme.fonts.mono,
              fontSize: theme.size.tiny,
              letterSpacing: 3,
              textTransform: "uppercase",
              color: colour,
              marginBottom: theme.space(2),
            }}
          >
            {label}
          </div>
          <div
            style={{
              fontSize: theme.size.heading,
              fontWeight: 650,
              lineHeight: 1.2,
              marginBottom: theme.space(2.5),
            }}
          >
            {heading}
          </div>
          <div style={{ fontSize: theme.size.small, lineHeight: 1.5, color: theme.colors.inkMuted }}>
            {body}
          </div>
          {code ? (
            <pre
              style={{
                marginTop: theme.space(3),
                marginBottom: 0,
                fontFamily: theme.fonts.mono,
                fontSize: theme.size.code,
                color: theme.colors.accent,
                background: theme.colors.bgPanelAlt,
                borderRadius: 10,
                padding: theme.space(2.5),
                whiteSpace: "pre-wrap",
              }}
            >
              {code}
            </pre>
          ) : null}
        </div>
      </div>
      <Disclaimer />
    </Stage>
  );
};
