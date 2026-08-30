# Narration pipeline

```
script.md ──parse──► sentences ──Kokoro-82M (ONNX, CPU)──► audio/*.wav
                                                              │
                                                              ▼
                                                        timings.json
                                                              │
                                          Remotion reads it ──┘  (scene durations,
                                                                  audio offsets,
                                                                  caption cues)
```

## Run it

```bash
bash video/kokoro/fetch_model.sh                       # once, ~350 MB
python3 video/kokoro/narrate.py --check --all          # parse every script, synthesise nothing
python3 video/kokoro/narrate.py video/topics/p0.1/script.md
python3 video/kokoro/narrate.py --all                  # everything, cached
python3 video/kokoro/narrate.py --all --force          # ignore the cache
```

## The script format

`script_format.py` has the full grammar; the short version is that a script is ordinary
markdown with two conventions:

* a scene heading is `## <id> · <ComponentName>` — the component name must exist in
  `video/src/Episode.tsx`'s `COMPONENTS` map;
* a fenced ```` ```props ```` block carries that component's props as JSON.

Everything else is narration. Sentences are the unit of synthesis, so write short declarative
sentences and end them with a full stop — the parser splits on sentence boundaries, and each
sentence becomes one wav, one caption cue, and one timing entry.

A scene with no narration (a `Transition`) needs `"hold": <seconds>` in its props.

## What `timings.json` guarantees

* `scenes[].start` / `.duration` — what Remotion uses for `<Sequence>`; the sum of the scene's
  sentence durations plus pauses, floored at the series' 8-second minimum.
* `segments[].file` — a path relative to the Remotion public dir (`video/topics`), so
  `staticFile("p0.1/audio/s1-000-<hash>.wav")` resolves.
* `wordsPerMinute` — the achieved pace. `narrate.py` warns outside 110–145.

## What is committed

`script.md`, `timings.json`, `audio/manifest.json` and `scenes.tsx` are committed. The `.wav`
files and the rendered `.mp4` are not — they are regenerable from the script, and the manifest
records the exact text and duration behind every one of them, so a reviewer can tell whether a
committed timing matches the committed script without downloading 400 MB of audio.
