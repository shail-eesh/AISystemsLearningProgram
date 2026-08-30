# Video production

Every topic ships a mini-course of episodes. This directory is the machinery that makes ~170 of
them look and sound like one series instead of 170 separate attempts.

```
script.md ──► narrate.py (Kokoro-82M, CPU) ──► audio/*.wav + timings.json
                                                          │
                    scenes.tsx + src/components/  ◄────────┘  durations, offsets, caption cues
                                                          │
                                              remotion render (Chromium, 720p)
                                                          │
                                                    out/<id>-e1.mp4   (delivered in chat)
```

## The one idea

**Narration drives duration.** A scene lasts exactly as long as its synthesised audio. Nothing is
ever hand-synced; rewrite a sentence, re-run `narrate.py`, and the whole episode re-times itself.
That single decision is what makes a 170-episode series maintainable by a fresh session each day.

## Layout

| path | what it is | committed? |
|---|---|---|
| `src/theme.ts` | palette, type, spacing — the series' entire visual identity | yes |
| `src/components/` | the ten reusable scene components | yes |
| `src/Episode.tsx` | turns a `timings.json` into a composition | yes |
| `src/Root.tsx` | every episode, registered | yes |
| `kokoro/` | the narration pipeline + script format | yes |
| `VOICE.md` | the locked narrator voice and settings | yes |
| `topics/<id>/script.md` | narration + scene directions | yes |
| `topics/<id>/timings.json` | generated durations and offsets | yes |
| `topics/<id>/scenes.tsx` | the composition registration | yes |
| `topics/<id>/audio/manifest.json` | text + duration behind every wav | yes |
| `topics/<id>/audio/*.wav` | the narration itself (~50 MB/episode) | **no** |
| `topics/<id>/out/*.mp4` | the render | **no** |

The wavs and mp4s are regenerable from what *is* committed, and the manifest records the exact
text and duration behind every wav — so a reviewer can check that a committed timing matches the
committed script without downloading half a gigabyte of audio.

## The component library

`TitleCard` · `ConceptScene` · `MathReveal` · `CodeWalkthrough` · `DiagramScene` · `ChartScene` ·
`ArchitectureMap` · `RecapScene` · `Callout` · `Transition`, plus `Captions`.

Three of them carry the pacing rules from MASTER_PLAN §5.1 directly:

* **`MathReveal`** enforces the three-way reveal — plain English, then symbols, then the line of
  code — so a symbol is never on screen before its meaning. Each stage has its own frame offset,
  tied to the narration.
* **`CodeWalkthrough`** dims every line except the active one and reveals long files
  progressively, driven by a `highlights` list of `{at, lines, caption}`.
* **`ArchitectureMap`** is the recap bookend: the AlphaDesk block diagram with the current
  topic lit up. The map is defined once, in that file, so "here's where we are in the build"
  means the same thing in all 51 topics.

Charts and diagrams are hand-drawn SVG rather than a charting library — one pinned dependency
across 170 episodes and three weeks of generation is a liability, and these shapes are twenty
lines each.

## Running it

```bash
cd video && npm install                       # once
bash kokoro/fetch_model.sh                    # once, ~350 MB, not committed
pip install kokoro-onnx soundfile

python3 kokoro/narrate.py --check --all       # validate every script, synthesise nothing
python3 kokoro/narrate.py --all               # synthesise (cached per sentence)
npx remotion compositions src/index.ts        # list what can be rendered
bash topics/p0.1/render.sh                    # render one episode
npm run studio                                # interactive preview

bash ../scripts/compress_episode.sh topics/p0.1/out/p0-1-e1.mp4   # delivery copy
```

**Chromium.** `remotion.media` is not on this sandbox's network allowlist, so Remotion cannot
download its own Chrome Headless Shell. `remotion.config.ts` points `setBrowserExecutable` at the
preinstalled Playwright Chromium; override it with `FORGE_CHROMIUM` on any other machine.

**Delivery size.** Remotion's default H.264 output runs ~3.7 MB per minute, which puts a
fifteen-minute episode over the 30 MiB chat upload limit. `scripts/compress_episode.sh` re-encodes
at CRF 23 with mono 64 kbps audio — these slides are static for seconds at a time, so it lands at
about a third of the size with no visible loss. The master render is left untouched.

## Known deviation (Day 1)

The Phase 0 episodes run **14–16 minutes**, against the 6–12 minute guideline in MASTER_PLAN
§5.1. They were written as single episodes per topic to match the Day 1 plan's "3 episodes", and
Phase 0's translation-heavy material did not compress into twelve minutes without cutting
material that belongs in the ramp.

The fix is to split them, and the pipeline change it needs is small: `narrate.py` currently
derives `timings.json` from the script's parent directory, so a topic can hold only one script.
Supporting `script-e1.md` → `timings-e1.json` is a few lines, and Day 2 should do it before
writing T31's five-episode mini-course, where the guideline matters much more.

Pacing itself is on target: the achieved narration rate is **129 words per minute**, inside the
120–135 band, and `narrate.py` warns whenever a script drifts outside 110–145.
