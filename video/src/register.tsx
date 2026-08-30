/** The one place a composition's frame rate, size and duration are decided. */

import React from "react";
import { Composition } from "remotion";
import { Episode, episodeFrames, type Timings } from "./Episode";
import { theme } from "./theme";

export type EpisodeSpec = {
  /** Composition id, e.g. "p0-1-e1". Remotion ids may not contain dots. */
  id: string;
  timings: Timings;
  captions?: boolean;
};

export const EpisodeComposition: React.FC<{ spec: EpisodeSpec }> = ({ spec }) => (
  <Composition
    id={spec.id}
    component={Episode}
    durationInFrames={episodeFrames(spec.timings)}
    fps={theme.video.fps}
    width={theme.video.width}
    height={theme.video.height}
    defaultProps={{ timings: spec.timings, captions: spec.captions ?? true }}
  />
);
