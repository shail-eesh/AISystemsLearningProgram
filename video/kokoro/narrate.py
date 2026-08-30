#!/usr/bin/env python3
"""Kokoro-82M narration pipeline: script.md -> audio/*.wav + timings.json.

    python3 video/kokoro/narrate.py video/topics/p0.1/script.md
    python3 video/kokoro/narrate.py --all

Why this exists in the shape it does:

* **Narration drives duration.** Every scene's length is the sum of its
  synthesised segment lengths plus the pauses. Remotion reads `timings.json`
  and lays scenes out accordingly, so nothing is ever hand-synced. Edit a
  sentence, re-run, and the whole episode re-times itself.
* **Sentences are the unit.** One wav per sentence means a caption cue per
  sentence for free, and it means a re-run only re-synthesises what changed
  (segments are content-addressed by a hash of voice + speed + text).
* **Pacing is enforced, not hoped for.** Speed 0.85 with explicit inter-sentence
  pauses lands at roughly 120-135 words per minute, the unhurried lecture pace
  the course asks for. The script reports the achieved rate so a drift is
  visible.

Model weights (Apache-2.0) are not committed. Fetch them once:

    bash video/kokoro/fetch_model.sh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_format import Script, parse_script  # noqa: E402

VIDEO_DIR = Path(__file__).resolve().parent.parent
TOPICS_DIR = VIDEO_DIR / "topics"
DEFAULT_MODEL_DIR = Path(os.environ.get("FORGE_KOKORO_DIR", Path.home() / ".cache/forge/kokoro"))
SAMPLE_RATE = 24_000

#: Pause inserted after each sentence, and the longer one after a scene.
PAUSE_SENTENCE = 0.42
PAUSE_SCENE = 0.65
#: MASTER_PLAN §5.1 — no scene shorter than this.
MIN_SCENE_SECONDS = 8.0


def model_paths(model_dir: Path | None = None) -> tuple[Path, Path]:
    d = model_dir or DEFAULT_MODEL_DIR
    model, voices = d / "kokoro-v1.0.onnx", d / "voices-v1.0.bin"
    if not model.exists() or not voices.exists():
        raise FileNotFoundError(
            f"Kokoro weights not found in {d}. Run `bash video/kokoro/fetch_model.sh` "
            "(or set FORGE_KOKORO_DIR)."
        )
    return model, voices


def load_kokoro(model_dir: Path | None = None):
    from kokoro_onnx import Kokoro

    model, voices = model_paths(model_dir)
    return Kokoro(str(model), str(voices))


def segment_key(text: str, voice: str, speed: float) -> str:
    payload = f"{voice}|{speed:.3f}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def synthesise_script(
    script: Script,
    out_dir: Path,
    model_dir: Path | None = None,
    force: bool = False,
) -> dict:
    """Render every sentence to a wav and return the timings document."""
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audio_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    kokoro = None
    cursor = 0.0
    scenes_out = []
    synthesised = reused = 0
    t0 = time.perf_counter()

    for scene in script.scenes:
        sentences = scene.sentences()
        scene_start = cursor
        segments = []

        for idx, sentence in enumerate(sentences):
            key = segment_key(sentence, script.voice, script.speed)
            name = f"{scene.id}-{idx:03d}-{key}.wav"
            path = audio_dir / name
            if path.exists() and not force and manifest.get(name, {}).get("text") == sentence:
                duration = float(manifest[name]["duration"])
                reused += 1
            else:
                if kokoro is None:
                    kokoro = load_kokoro(model_dir)
                samples, sr = kokoro.create(
                    sentence, voice=script.voice, speed=script.speed, lang="en-us"
                )
                if sr != SAMPLE_RATE:
                    raise RuntimeError(f"unexpected sample rate {sr}")
                sf.write(path, samples, sr)
                duration = len(samples) / sr
                manifest[name] = {"text": sentence, "duration": duration, "voice": script.voice}
                synthesised += 1

            segments.append(
                {
                    "text": sentence,
                    "file": f"{script.topic}/audio/{name}",
                    "start": round(cursor, 4),
                    "duration": round(duration, 4),
                }
            )
            cursor += duration + PAUSE_SENTENCE

        if segments:
            cursor += PAUSE_SCENE - PAUSE_SENTENCE
        else:
            cursor += float(scene.hold or MIN_SCENE_SECONDS)

        duration = cursor - scene_start
        if duration < MIN_SCENE_SECONDS:
            cursor = scene_start + MIN_SCENE_SECONDS
            duration = MIN_SCENE_SECONDS

        scenes_out.append(
            {
                "id": scene.id,
                "component": scene.component,
                "props": scene.props,
                "start": round(scene_start, 4),
                "duration": round(duration, 4),
                "segments": segments,
            }
        )

    # Prune manifest entries whose wav is gone (a re-worded script).
    live = {seg["file"].split("/")[-1] for s in scenes_out for seg in s["segments"]}
    for stale in [k for k in manifest if k not in live]:
        manifest.pop(stale)
        (audio_dir / stale).unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    total = round(cursor, 4)
    words = script.word_count
    timings = {
        "topic": script.topic,
        "episode": script.episode,
        "title": script.title,
        "voice": script.voice,
        "speed": script.speed,
        "sampleRate": SAMPLE_RATE,
        "totalSeconds": total,
        "wordCount": words,
        "wordsPerMinute": round(words / (total / 60.0), 1) if total else 0.0,
        "sceneCount": len(scenes_out),
        "shortestSceneSeconds": round(min(s["duration"] for s in scenes_out), 2),
        "generatedInSeconds": round(time.perf_counter() - t0, 1),
        "segmentsSynthesised": synthesised,
        "segmentsReused": reused,
        "scenes": scenes_out,
    }
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=1) + "\n")
    return timings


def report(timings: dict) -> None:
    mins, secs = divmod(timings["totalSeconds"], 60)
    print(
        f"  {timings['topic']} E{timings['episode']}: {int(mins)}m{secs:04.1f}s · "
        f"{timings['sceneCount']} scenes · {timings['wordCount']} words · "
        f"{timings['wordsPerMinute']} wpm · shortest scene "
        f"{timings['shortestSceneSeconds']}s"
    )
    print(
        f"    synthesised {timings['segmentsSynthesised']}, reused "
        f"{timings['segmentsReused']}, in {timings['generatedInSeconds']}s"
    )
    wpm = timings["wordsPerMinute"]
    if not 110 <= wpm <= 145:
        print(f"    ! pacing outside the 120-135 wpm target ({wpm}); adjust speed or sentences")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthesise episode narration with Kokoro-82M.")
    ap.add_argument("script", nargs="?", help="path to a script.md")
    ap.add_argument("--all", action="store_true", help="every video/topics/*/script.md")
    ap.add_argument("--force", action="store_true", help="re-synthesise even if cached")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="parse only; do not synthesise")
    args = ap.parse_args(argv)

    if args.all:
        scripts = sorted(TOPICS_DIR.glob("*/script.md"))
    elif args.script:
        scripts = [Path(args.script)]
    else:
        ap.error("pass a script path or --all")

    if not scripts:
        print("no scripts found")
        return 1

    for path in scripts:
        script = parse_script(path)
        if args.check:
            words = script.word_count
            print(f"  {path.parent.name}: OK · {len(script.scenes)} scenes · {words} words")
            continue
        report(synthesise_script(script, path.parent, args.model_dir, args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
