"""Parser for the episode script format.

A `script.md` is a normal, readable markdown document that is also machine
parseable. That matters: the script is what a human edits when a sentence
lands badly, so it must not become JSON.

    ---
    topic: p0.1
    episode: 1
    title: Python for .NET architects
    voice: am_michael
    speed: 0.85
    ---

    ## s1 · TitleCard

    ```props
    {"title": "Python for the .NET veteran", "episode": "Episode 1"}
    ```

    Narration for this scene. Sentences are the unit of synthesis, so write
    them short and end them with a full stop.

Rules:

* Scene headings are `## <id> · <ComponentName>`. The id must be unique.
* A fenced ```props block carries the component's props as JSON.
* Everything else that is not a fenced block is narration prose.
* A scene with no narration is allowed (a Transition, say) and gets a
  duration from its `hold` prop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HEADING = re.compile(r"^##\s+(?P<id>[A-Za-z0-9_.-]+)\s*[·|-]\s*(?P<component>[A-Za-z]+)\s*$")
FENCE = re.compile(r"^```(?P<lang>[a-zA-Z]*)\s*$")
FRONTMATTER = re.compile(r"^---\s*$")
#: Sentence boundary that tolerates abbreviations and decimals.
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class ScriptError(ValueError):
    """The script does not follow the format."""


@dataclass
class Scene:
    id: str
    component: str
    props: dict = field(default_factory=dict)
    narration: str = ""
    #: Seconds to hold when there is no narration (transitions, stills).
    hold: float | None = None

    def sentences(self) -> list[str]:
        text = " ".join(self.narration.split())
        if not text:
            return []
        parts = [s.strip() for s in SENTENCE.split(text) if s.strip()]
        return parts or [text]


@dataclass
class Script:
    topic: str
    episode: int
    title: str
    voice: str
    speed: float
    scenes: list[Scene]
    meta: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return sum(len(s.narration.split()) for s in self.scenes)


def _split_frontmatter(lines: list[str]) -> tuple[dict, list[str]]:
    if not lines or not FRONTMATTER.match(lines[0]):
        raise ScriptError("script must start with a YAML frontmatter block")
    for i in range(1, len(lines)):
        if FRONTMATTER.match(lines[i]):
            meta = yaml.safe_load("\n".join(lines[1:i])) or {}
            return meta, lines[i + 1:]
    raise ScriptError("unterminated frontmatter block")


def parse_script(path: str | Path) -> Script:
    text = Path(path).read_text(encoding="utf-8")
    meta, body = _split_frontmatter(text.splitlines())

    for key in ("topic", "episode", "title"):
        if key not in meta:
            raise ScriptError(f"frontmatter is missing '{key}'")

    scenes: list[Scene] = []
    current: Scene | None = None
    fence_lang: str | None = None
    fence_buf: list[str] = []
    prose: list[str] = []

    def flush() -> None:
        if current is not None:
            current.narration = "\n".join(prose).strip()
            scenes.append(current)

    for raw in body:
        fence = FENCE.match(raw.rstrip())
        if fence and fence_lang is None:
            fence_lang = fence.group("lang") or "text"
            fence_buf = []
            continue
        if raw.rstrip() == "```" and fence_lang is not None:
            if fence_lang == "props":
                if current is None:
                    raise ScriptError("a props block appeared before any scene heading")
                try:
                    current.props = json.loads("\n".join(fence_buf))
                except json.JSONDecodeError as exc:
                    raise ScriptError(f"scene {current.id}: props is not valid JSON: {exc}") from exc
                current.hold = current.props.pop("hold", None)
            fence_lang = None
            continue
        if fence_lang is not None:
            fence_buf.append(raw)
            continue

        heading = HEADING.match(raw.rstrip())
        if heading:
            flush()
            prose = []
            current = Scene(id=heading.group("id"), component=heading.group("component"))
            continue
        if current is not None:
            prose.append(raw)

    flush()

    if not scenes:
        raise ScriptError("no scenes found; headings look like '## s1 · ConceptScene'")
    ids = [s.id for s in scenes]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ScriptError(f"duplicate scene ids: {sorted(dupes)}")
    for s in scenes:
        if not s.sentences() and s.hold is None:
            raise ScriptError(f"scene {s.id} has neither narration nor a 'hold' in its props")

    return Script(
        topic=str(meta["topic"]),
        episode=int(meta["episode"]),
        title=str(meta["title"]),
        voice=str(meta.get("voice", "am_michael")),
        speed=float(meta.get("speed", 0.85)),
        scenes=scenes,
        meta=meta,
    )
