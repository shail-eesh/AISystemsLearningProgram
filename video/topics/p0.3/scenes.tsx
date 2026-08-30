/**
 * P0.3 episode compositions.
 *
 * The scenes themselves live in `script.md` (component + props per scene) and
 * their durations come from `timings.json`, which `video/kokoro/narrate.py`
 * regenerates from the narration audio. This file only names the composition,
 * so a re-worded script never needs a code change.
 */

import type { Timings } from "../../src/Episode";
import type { EpisodeSpec } from "../../src/register";
import timings from "./timings.json";

export const P0_3_EPISODES: EpisodeSpec[] = [
  { id: "p0-3-e1", timings: timings as unknown as Timings, captions: true },
];
