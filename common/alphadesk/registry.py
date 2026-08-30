"""The component registry topics plug into.

Why a registry at all? Because 51 topics built over 15 runs cannot import each
other directly without turning into a dependency knot. Instead every topic
*declares* itself — id, surface, capability, a factory — and AlphaDesk composes
whatever happens to be present. Missing topics degrade the desk, they never
break it.

.NET analogy: this is a hand-rolled `IServiceCollection` with attributes for
registration and a very small resolver. Python has no built-in DI container and
does not need one; a dict plus a decorator gets you 95% of the value.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Surface(str, Enum):
    """The three product surfaces from MASTER_PLAN §6.1, plus shared layers."""

    RESEARCH = "research"        # Research Copilot: RAG, KG, reasoning
    ORDERS = "orders"            # Order Workflow (paper): tickets, DSL, risk
    COMPLIANCE = "compliance"    # Guardrails, eval, surveillance, gateway
    MODELS = "models"            # tokenizer, SLM, embeddings, serving
    DATA = "data"                # loaders, curation, feature store
    FOUNDATION = "foundation"    # Phase 0/1 building blocks


@dataclass(frozen=True)
class Component:
    """One topic's contribution to the desk."""

    topic: str                       # e.g. "T30"
    name: str                        # e.g. "fintok"
    surface: Surface
    summary: str
    factory: Callable[..., Any]
    module: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.surface.value}.{self.name}"

    def build(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the component. Kept lazy so import cost stays near zero."""
        return self.factory(*args, **kwargs)


class DuplicateComponent(KeyError):
    """Two topics claimed the same surface.name key."""


class Registry:
    """An ordered, introspectable collection of components."""

    def __init__(self) -> None:
        self._items: dict[str, Component] = {}

    # -- registration -----------------------------------------------------
    def add(self, component: Component, *, replace: bool = False) -> Component:
        if component.key in self._items and not replace:
            raise DuplicateComponent(
                f"{component.key} already registered by topic "
                f"{self._items[component.key].topic}"
            )
        self._items[component.key] = component
        return component

    # -- lookup -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self):
        return iter(self._items.values())

    def get(self, key: str) -> Component:
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(f"no component {key!r}; have {sorted(self._items)}") from exc

    def by_topic(self, topic: str) -> list[Component]:
        return [c for c in self._items.values() if c.topic.upper() == topic.upper()]

    def by_surface(self, surface: Surface | str) -> list[Component]:
        s = Surface(surface) if not isinstance(surface, Surface) else surface
        return [c for c in self._items.values() if c.surface is s]

    def topics(self) -> list[str]:
        return sorted({c.topic for c in self._items.values()})

    def missing_requirements(self) -> dict[str, list[str]]:
        """Components whose declared prerequisites are not registered yet."""
        present = {c.topic.upper() for c in self._items.values()}
        gaps = {}
        for c in self._items.values():
            absent = [r for r in c.requires if r.upper() not in present]
            if absent:
                gaps[c.key] = absent
        return gaps

    def clear(self) -> None:
        """Test hook — the registry is process-global."""
        self._items.clear()

    # -- discovery --------------------------------------------------------
    def load_modules(self, modules: Iterable[str]) -> dict[str, str]:
        """Import each module so its @register decorators run.

        Returns a map of module -> error string for the ones that failed, so a
        half-built desk still starts. This is the whole reason topics never
        import each other directly.

        The import happens with `self` installed as the *active* registry, and
        an already-imported module is reloaded — otherwise `sys.modules`
        caching would mean a second registry never receives anything.
        """
        errors: dict[str, str] = {}
        with self.activate():
            for m in modules:
                try:
                    if m in sys.modules:
                        importlib.reload(sys.modules[m])
                    else:
                        importlib.import_module(m)
                except Exception as exc:  # noqa: BLE001 - deliberate: degrade, don't die
                    errors[m] = f"{type(exc).__name__}: {exc}"
        return errors

    @contextmanager
    def activate(self):
        """Make this the registry that bare @register calls file into."""
        _ACTIVE.append(self)
        try:
            yield self
        finally:
            _ACTIVE.pop()

    def describe(self) -> str:
        """Human-readable desk state — what the Phase 9 demo prints on boot."""
        if not self._items:
            return "AlphaDesk: no components registered yet."
        lines = [f"AlphaDesk: {len(self._items)} component(s) across {len(self.topics())} topic(s)"]
        for surface in Surface:
            items = self.by_surface(surface)
            if not items:
                continue
            lines.append(f"  {surface.value}:")
            for c in sorted(items, key=lambda x: x.name):
                lines.append(f"    - {c.name:<22} [{c.topic}] {c.summary}")
        gaps = self.missing_requirements()
        if gaps:
            lines.append("  unmet requirements:")
            for k, v in sorted(gaps.items()):
                lines.append(f"    - {k} needs {', '.join(v)}")
        return "\n".join(lines)


REGISTRY = Registry()

#: Stack of registries; the top one receives bare @register calls.
_ACTIVE: list[Registry] = []


def current_registry() -> Registry:
    """The registry a bare @register targets right now."""
    return _ACTIVE[-1] if _ACTIVE else REGISTRY


def register(
    *,
    topic: str,
    name: str,
    surface: Surface | str,
    summary: str,
    requires: Iterable[str] = (),
    registry: Registry | None = None,
    replace: bool = False,
):
    """Decorator that files a class or factory function into the registry."""

    def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
        component = Component(
            topic=topic,
            name=name,
            surface=Surface(surface) if not isinstance(surface, Surface) else surface,
            summary=summary,
            factory=obj,
            module=getattr(obj, "__module__", ""),
            requires=tuple(requires),
        )
        # NOTE: `registry or current_registry()` would be a bug — an empty
        # Registry is falsy because __len__ returns 0, so a caller's fresh
        # registry would be silently swapped out. Identity check, always.
        target = current_registry() if registry is None else registry
        target.add(component, replace=replace)
        obj.__alphadesk__ = component  # type: ignore[attr-defined]
        return obj

    return decorator
