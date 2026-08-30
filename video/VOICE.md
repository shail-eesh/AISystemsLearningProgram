# Series narrator voice

One voice, one set of settings, for all ~170 episodes. Consistency across a course generated
over weeks by separate sessions is worth more than picking the "best" voice for any one episode,
so this file is the lock.

## The pick

| setting | value | why |
|---|---|---|
| model | **Kokoro-82M** (`kokoro-v1.0.onnx`) | 82M params, Apache-2.0 weights, real-time+ on container CPU via ONNX Runtime |
| voice | **`am_michael`** | American English, male, even cadence and low sibilance — it holds up over a ten-minute technical read where brighter voices get fatiguing |
| speed | **0.85** | the default rate is an explainer-video pace; 0.85 lands in the 120–135 wpm lecture band the course specifies |
| language | `en-us` | |
| sample rate | 24 000 Hz mono | Kokoro's native output; no resampling anywhere in the pipeline |
| sentence pause | 0.42 s | inserted between sentences, not baked into the audio |
| scene pause | 0.65 s | after the last sentence of a scene |

Measured on this container: ~2.3× real time (a 4.4-second sentence synthesises in about 1.9 s
cold, faster warm), so a ten-minute episode's narration is a few minutes of CPU.

## Why the pauses are not in the audio

Kokoro renders one wav per **sentence**, and the silences are inserted by the timing arithmetic
in `narrate.py`. Three things fall out of that:

* captions get a cue per sentence for free;
* re-running after an edit re-synthesises only the sentences that actually changed (segments are
  content-addressed by `sha256(voice | speed | text)`);
* pacing is a number you can tune in one place rather than a property baked into 900 wav files.

## Changing the voice

Don't, unless a whole re-render is planned. If it must change: edit the `voice:` line in every
`video/topics/*/script.md` frontmatter, update this file, and re-run `narrate.py --all --force`.
The hashes change, so every segment re-synthesises. Episodes rendered before and after will not
match, which is exactly the drift this file exists to prevent.

## Weights

Not committed (~350 MB). Fetch once:

```bash
bash video/kokoro/fetch_model.sh          # -> ~/.cache/forge/kokoro
pip install kokoro-onnx soundfile
```

Kokoro-82M weights are Apache-2.0. The synthesised narration in this course is generated from
scripts written for it; no third-party audio is redistributed.
