/**
 * Every episode in the series, registered as a Remotion composition.
 *
 * One line per episode. Each topic's `scenes.tsx` exports its compositions;
 * this file only collects them, so a daily run adds an import and an entry.
 */

import React from "react";
import { Folder } from "remotion";
import { P0_1_EPISODES } from "../topics/p0.1/scenes";
import { P0_2_EPISODES } from "../topics/p0.2/scenes";
import { P0_3_EPISODES } from "../topics/p0.3/scenes";
import { EpisodeComposition, type EpisodeSpec } from "./register";

const PHASE_0: EpisodeSpec[] = [...P0_1_EPISODES, ...P0_2_EPISODES, ...P0_3_EPISODES];

export const RemotionRoot: React.FC = () => (
  <>
    <Folder name="Phase-0">
      {PHASE_0.map((spec) => (
        <EpisodeComposition key={spec.id} spec={spec} />
      ))}
    </Folder>
  </>
);
