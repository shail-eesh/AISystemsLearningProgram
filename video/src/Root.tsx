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
import { T15_E1_EPISODES } from "../topics/t15-e1/scenes";
import { T15_E2_EPISODES } from "../topics/t15-e2/scenes";
import { T15_E3_EPISODES } from "../topics/t15-e3/scenes";
import { T15_E4_EPISODES } from "../topics/t15-e4/scenes";
import { T4_E1_EPISODES } from "../topics/t4-e1/scenes";
import { T4_E2_EPISODES } from "../topics/t4-e2/scenes";
import { T4_E3_EPISODES } from "../topics/t4-e3/scenes";
import { T4_E4_EPISODES } from "../topics/t4-e4/scenes";
import { T4_E5_EPISODES } from "../topics/t4-e5/scenes";
import { T16A_E1_EPISODES } from "../topics/t16a-e1/scenes";
import { T16A_E2_EPISODES } from "../topics/t16a-e2/scenes";
import { T30_E1_EPISODES } from "../topics/t30-e1/scenes";
import { T30_E2_EPISODES } from "../topics/t30-e2/scenes";
import { T30_E3_EPISODES } from "../topics/t30-e3/scenes";
import { T31_E1_EPISODES } from "../topics/t31-e1/scenes";
import { T31_E2_EPISODES } from "../topics/t31-e2/scenes";
import { T31_E3_EPISODES } from "../topics/t31-e3/scenes";
import { T45A_E1_EPISODES } from "../topics/t45a-e1/scenes";
import { T45A_E2_EPISODES } from "../topics/t45a-e2/scenes";
import { EpisodeComposition, type EpisodeSpec } from "./register";

const PHASE_0: EpisodeSpec[] = [...P0_1_EPISODES, ...P0_2_EPISODES, ...P0_3_EPISODES];

const PHASE_1: EpisodeSpec[] = [
  ...T31_E1_EPISODES,
  ...T31_E2_EPISODES,
  ...T31_E3_EPISODES,
  ...T16A_E1_EPISODES,
  ...T16A_E2_EPISODES,
  ...T45A_E1_EPISODES,
  ...T45A_E2_EPISODES,
  ...T30_E1_EPISODES,
  ...T30_E2_EPISODES,
  ...T30_E3_EPISODES,
];

const PHASE_2: EpisodeSpec[] = [
  ...T4_E1_EPISODES,
  ...T4_E2_EPISODES,
  ...T4_E3_EPISODES,
  ...T4_E4_EPISODES,
  ...T4_E5_EPISODES,
  ...T15_E1_EPISODES,
  ...T15_E2_EPISODES,
  ...T15_E3_EPISODES,
  ...T15_E4_EPISODES,
];

export const RemotionRoot: React.FC = () => (
  <>
    <Folder name="Phase-0">
      {PHASE_0.map((spec) => (
        <EpisodeComposition key={spec.id} spec={spec} />
      ))}
    </Folder>
    <Folder name="Phase-1">
      {PHASE_1.map((spec) => (
        <EpisodeComposition key={spec.id} spec={spec} />
      ))}
    </Folder>
    <Folder name="Phase-2">
      {PHASE_2.map((spec) => (
        <EpisodeComposition key={spec.id} spec={spec} />
      ))}
    </Folder>
  </>
);
