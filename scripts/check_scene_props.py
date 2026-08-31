#!/usr/bin/env python3
"""Validate every episode script's scene props before a render is attempted.

    python3 scripts/check_scene_props.py            # all scripts
    python3 scripts/check_scene_props.py video/topics/t31-e3/script.md

Why this exists: `narrate.py --check` validates the *script grammar* and is
happy with any JSON in a ```props block. Remotion then bundles fine and fails
at the frame where the bad prop is first read — twelve minutes into a render,
with a stack trace inside `scheduler.production.min.js` that names neither the
scene nor the prop.

That happened on Day 2: four line charts were written with `points: [[x, y],
...]` when `ChartScene`'s `Series` type wants `values: [y, ...]` (the x axis is
positional). Three renders died after the money had been spent. This script
turns that into a one-second failure with the scene id in the message.

It is a deliberately shallow contract check, not a type system: required keys,
key types, and the handful of cross-field rules that have actually bitten.
"""

from __future__ import annotations

import pathlib
import sys

VIDEO = pathlib.Path(__file__).resolve().parent.parent / "video"
SRC = VIDEO / "src" / "components"

#: component -> (required props, optional props)
CONTRACT: dict[str, tuple[set[str], set[str]]] = {
    "TitleCard": ({"title"}, {"subtitle", "topicId", "episode", "paper"}),
    "ConceptScene": ({"title"}, {"eyebrow", "body", "points", "aside"}),
    "MathReveal": (
        {"title", "english", "equation", "code"},
        {"eyebrow", "note", "stageFrames"},
    ),
    "CodeWalkthrough": (
        {"title", "code"},
        {"eyebrow", "filename", "highlights", "revealStart", "revealEvery", "fontSize"},
    ),
    "DiagramScene": ({"title", "nodes"}, {"eyebrow", "edges", "caption"}),
    "ChartScene": ({"title", "kind"}, {"eyebrow", "series", "bars", "yLabel", "xLabel",
                                       "reference", "caption", "logScale"}),
    "ArchitectureMap": ({"highlight"}, {"eyebrow", "title", "caption"}),
    "RecapScene": ({"points", "ifSkipped"}, {"eyebrow", "title", "next"}),
    "Callout": ({"heading", "body"}, {"kind", "code"}),
    "Transition": (set(), {"label", "hold"}),
}

CALLOUT_KINDS = {"gotcha", "warning", "insight"}


def architecture_block_ids() -> set[str]:
    """Read the block ids straight out of the component, so the two cannot drift."""
    text = (SRC / "ArchitectureMap.tsx").read_text()
    return {line.split('id: "')[1].split('"')[0] for line in text.splitlines() if 'id: "' in line}


def check_scene(scene, blocks: set[str]) -> list[str]:
    where = f"{scene.id} ({scene.component})"
    if scene.component not in CONTRACT:
        return [f"{where}: unknown component; known: {', '.join(sorted(CONTRACT))}"]

    required, optional = CONTRACT[scene.component]
    props = scene.props
    problems = [f"{where}: missing required prop '{k}'" for k in sorted(required - set(props))]
    problems += [
        f"{where}: unknown prop '{k}' (typo? known: {', '.join(sorted(required | optional))})"
        for k in sorted(set(props) - required - optional)
    ]

    if scene.component == "ChartScene":
        kind = props.get("kind")
        if kind not in ("line", "bar"):
            problems.append(f"{where}: kind must be 'line' or 'bar', got {kind!r}")
        if kind == "line":
            for i, s in enumerate(props.get("series") or []):
                if "values" not in s:
                    problems.append(
                        f"{where}: series[{i}] has no 'values' — the x axis is positional, "
                        "so a line series is a flat list of y values"
                    )
                elif not all(isinstance(v, int | float) for v in s["values"]):
                    problems.append(f"{where}: series[{i}].values must be plain numbers")
            if not props.get("series"):
                problems.append(f"{where}: kind='line' needs a non-empty 'series'")
            if props.get("logScale"):
                problems.append(f"{where}: logScale is only implemented for bar charts")
        if kind == "bar":
            for i, b in enumerate(props.get("bars") or []):
                if "value" not in b or "label" not in b:
                    problems.append(f"{where}: bars[{i}] needs 'label' and 'value'")
            if not props.get("bars"):
                problems.append(f"{where}: kind='bar' needs a non-empty 'bars'")

    if scene.component == "DiagramScene":
        ids = {n.get("id") for n in props.get("nodes") or []}
        for e in props.get("edges") or []:
            for end in ("from", "to"):
                if e.get(end) not in ids:
                    problems.append(f"{where}: edge {end}={e.get(end)!r} is not a node id")

    if scene.component == "ArchitectureMap":
        for h in props.get("highlight") or []:
            if h not in blocks:
                problems.append(
                    f"{where}: highlight {h!r} is not an AlphaDesk block; "
                    f"have {', '.join(sorted(blocks))}"
                )

    if scene.component == "Callout":
        kind = props.get("kind", "gotcha")
        if kind not in CALLOUT_KINDS:
            problems.append(f"{where}: kind must be one of {sorted(CALLOUT_KINDS)}, got {kind!r}")

    if scene.component == "MathReveal":
        stages = props.get("stageFrames")
        if stages is not None and (len(stages) != 3 or sorted(stages) != list(stages)):
            problems.append(f"{where}: stageFrames must be three ascending frame numbers")

    return problems


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(VIDEO / "kokoro"))
    from script_format import parse_script  # noqa: PLC0415

    paths = [pathlib.Path(a) for a in argv[1:]] or sorted((VIDEO / "topics").glob("*/script.md"))
    blocks = architecture_block_ids()
    failures = 0
    for path in paths:
        script = parse_script(path)
        problems = [p for scene in script.scenes for p in check_scene(scene, blocks)]
        if problems:
            failures += 1
            print(f"  {path.parent.name}: {len(problems)} problem(s)")
            for p in problems:
                print(f"      {p}")
        else:
            print(f"  {path.parent.name}: OK · {len(script.scenes)} scenes")
    print(f"\n-> {'PASS' if not failures else f'FAIL — {failures} script(s)'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
