/**
 * The composer. Turns a `timings.json` produced by the Kokoro pipeline into a
 * rendered episode.
 *
 * The one idea that makes this maintainable: **narration drives duration.**
 * Each scene lasts exactly as long as its synthesised audio, so nobody ever
 * hand-syncs a slide to a voice-over. Rewrite a sentence, re-run `narrate.py`,
 * and every downstream timing moves by itself.
 */

import React from "react";
import { AbsoluteFill, Audio, Sequence, staticFile, useVideoConfig } from "remotion";
import {
  ArchitectureMap,
  Callout,
  Captions,
  ChartScene,
  CodeWalkthrough,
  ConceptScene,
  DiagramScene,
  MathReveal,
  RecapScene,
  TitleCard,
  Transition,
} from "./components";
import type { CaptionCue } from "./components/Captions";
import { theme } from "./theme";

/* eslint-disable @typescript-eslint/no-explicit-any */
export const COMPONENTS: Record<string, React.FC<any>> = {
  TitleCard,
  ConceptScene,
  MathReveal,
  CodeWalkthrough,
  DiagramScene,
  ChartScene,
  ArchitectureMap,
  RecapScene,
  Callout,
  Transition,
};

export type Segment = {
  text: string;
  file: string;
  start: number;
  duration: number;
};

export type SceneTiming = {
  id: string;
  component: string;
  props: Record<string, unknown>;
  start: number;
  duration: number;
  segments: Segment[];
};

export type Timings = {
  topic: string;
  episode: number;
  title: string;
  voice: string;
  speed: number;
  totalSeconds: number;
  scenes: SceneTiming[];
};

export type EpisodeProps = {
  timings: Timings;
  captions?: boolean;
};

const cuesFrom = (timings: Timings): CaptionCue[] =>
  timings.scenes.flatMap((s) =>
    s.segments.map((seg) => ({
      start: seg.start,
      end: seg.start + seg.duration,
      text: seg.text,
    })),
  );

export const Episode: React.FC<EpisodeProps> = ({ timings, captions = true }) => {
  const { fps } = useVideoConfig();
  const f = (seconds: number) => Math.max(1, Math.round(seconds * fps));

  return (
    <AbsoluteFill style={{ backgroundColor: theme.colors.bg }}>
      {timings.scenes.map((scene) => {
        const Component = COMPONENTS[scene.component];
        if (!Component) {
          throw new Error(
            `unknown component "${scene.component}" in ${timings.topic} scene ${scene.id}; ` +
              `known: ${Object.keys(COMPONENTS).join(", ")}`,
          );
        }
        return (
          <Sequence
            key={scene.id}
            name={`${scene.id} · ${scene.component}`}
            from={f(scene.start)}
            durationInFrames={f(scene.duration)}
          >
            <Component {...scene.props} />
          </Sequence>
        );
      })}

      {timings.scenes.flatMap((scene) =>
        scene.segments.map((seg) => (
          <Sequence
            key={seg.file}
            name={`audio ${seg.file.split("/").pop()}`}
            from={f(seg.start)}
            durationInFrames={f(seg.duration)}
          >
            <Audio src={staticFile(seg.file)} />
          </Sequence>
        )),
      )}

      {captions ? <Captions cues={cuesFrom(timings)} /> : null}
    </AbsoluteFill>
  );
};

/** Total frames an episode occupies, for `<Composition durationInFrames>`. */
export const episodeFrames = (timings: Timings, fps = theme.video.fps): number =>
  Math.max(1, Math.round(timings.totalSeconds * fps));
