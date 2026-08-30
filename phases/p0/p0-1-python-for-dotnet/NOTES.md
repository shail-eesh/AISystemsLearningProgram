# P0.1 · Python for the .NET veteran — Notes

*Intuition and gotchas. This file is the source the video script is written from.*

---

## The one-sentence version

Python gives you the same object-oriented vocabulary as C# with about a third of the ceremony,
and pays for it by moving four guarantees — immutability, interface conformance, numeric
exactness, and thread safety — from the compiler into your discipline.

## 1. Dataclasses are records with a shallower promise

`@dataclass(frozen=True)` buys you what `record` buys you: structural equality, a real
`__repr__`, a hash, and blocked attribute rebinding. The gap is depth.

```python
@dataclass(frozen=True)
class Order:
    fills: list[Fill] = field(default_factory=list)

o = Order()
o.fills = []          # FrozenInstanceError
o.fills.append(fill)  # perfectly legal
```

`frozen` protects the *reference*, not what it points at. C# `readonly` has the identical hole;
you notice it less there because `record struct` copies. Practical rule: freeze value objects,
leave lifecycle objects (`Order`) mutable and honest about it.

**The mutable-default trap has no C# analogue at all.** `def f(x, bucket=[])` evaluates that
list **once, at function definition**, and every call shares it. Dataclasses refuse a mutable
default outright, which is why `field(default_factory=list)` exists. In plain functions nothing
stops you; step 1 shows two calls returning the same list.

`__post_init__` runs *after* the fields are assigned, not before like a ctor guard. So validation
sees a fully-formed object — convenient, but it means a "half-valid" instance genuinely exists
for a moment, and in a frozen dataclass you must use `object.__setattr__` to normalise a field.

## 2. Money is `Decimal`, and `bool` is an `int`

You already know not to price with `double`. Same rule, same reason: `0.1 + 0.2 != 0.3` in IEEE
754 regardless of language. `Money` here refuses a `float` at the boundary rather than silently
quantizing it, because a wrong price that *looks* right is the expensive kind.

Two Python-specific landmines:

* `isinstance(True, int)` is `True`. `bool` subclasses `int`, so `Money(True)` would happily
  become ₹1 if you only check `isinstance(x, int)`. Check `bool` first.
* `round()` on floats does banker's rounding (`round(0.5) == 0`). Money wants
  `ROUND_HALF_UP`, which is a `Decimal` context argument — never `round()`.

**Where the storage precision came from.** The benchmark reconciles realised P&L against an
independent float reducer. At four decimal places of storage the worst relative error was
`2.8e-05` — thirty times the 1e-6 tolerance. Nothing was *wrong*: an average cost is a
quotient, and re-rounding it on every fill accumulates. Storing at six dp and displaying at four
dropped it to `3.1e-07`. That is the whole lesson of the topic in one number: **precision is a
property of the operation, not of the type.**

## 3. Enums carry behaviour; interfaces disappear

A C# enum is a named integer and behaviour arrives via extension methods. A Python `Enum` member
is an object, so `Side.BUY.sign` and `Side.BUY.opposite()` live where they belong. Subclassing
`str` additionally makes the member parse from and compare equal to the wire format — worth it
for anything crossing a JSON boundary, mildly dangerous if you rely on `is` in one place and
`==` in another.

`typing.Protocol` is the bigger shift. `IRiskCheck` does not exist; a class with a
`check(order)` method *is* a risk check. Nothing is declared, nothing is implemented, and a
mismatch surfaces at type-check time (or at call time, if you skipped the type checker). Coming
from .NET the instinct is to define the interface and inherit; resist it. Write the concrete
class, and add the `Protocol` only where a function needs to *state* what shape it accepts.

## 4. `NotImplemented` is not `NotImplementedError`

The single most surprising line in `money.py`:

```python
def __add__(self, other):
    if not isinstance(other, Money):
        return NotImplemented        # not raise!
```

Returning the `NotImplemented` **singleton** tells Python "I don't know this operand" — it then
tries `other.__radd__(self)` and only raises `TypeError` if that fails too. Raising
`NotImplementedError` instead permanently breaks `SomethingElse + Money`, and the error message
blames the wrong class. C# has no equivalent: operator resolution is static there.

## 5. Truthiness is a footgun with a real scar

`__bool__` means "is this meaningful", and empty collections, `0`, `Money(0)` and `Quantity(0)`
are all falsy. The idiom `x = arg or DEFAULT` therefore silently replaces *any* falsy argument,
not just `None`.

This bit the AlphaDesk registry on the day it was written. `(registry or REGISTRY).add(...)`
looked obviously correct — and quietly routed every registration to the global registry whenever
the caller passed a **fresh, empty** one, because `Registry.__len__` returns 0. The test failed
with `assert 0 == 1` and the fix was `REGISTRY if registry is None else registry`. Rule:
**`is None`, never `or`, for optional arguments.**

## 6. `with` is `using` with opinions

`__enter__`/`__exit__` is `IDisposable`, plus one power C# does not give you: `__exit__` receives
the in-flight exception and can **suppress** it by returning `True`. That makes rollback blocks
elegant and makes accidental exception-swallowing easy — so return `None` unless you mean it.

`@contextmanager` writes both halves from a single generator: everything before `yield` is
`__enter__`, everything after is `__exit__`, and `try/except/else/finally` around the `yield`
gives you rollback, commit, and cleanup in the shape you already think in.

## 7. Generators are `IEnumerable<T>`, and `itertools` is LINQ

A function with `yield` returns a lazy, **single-pass** iterator — that last part differs from
`IEnumerable<T>`, which you can usually enumerate twice. Once exhausted, a generator is done;
if you need to re-read, materialise it into a list deliberately.

The LINQ translation table is mostly `itertools` plus comprehensions:
`Select` → comprehension · `Where` → comprehension with `if` · `Take` → `islice` ·
`GroupBy` → `groupby` **on sorted input** (it groups *consecutive* keys — the number-one
`itertools` bug) · `Aggregate` → `functools.reduce` · `Any`/`All` → `any`/`all`.

## 8. async/await: same words, different machine

Five differences, in the order they hurt:

1. **Coroutines are cold.** `DoAsync()` in .NET is usually already running when you await it.
   `async def f()` returns an object that has done nothing. Forget the `await` and the work
   never happens — Python warns, but at *garbage-collection* time, which may be nowhere near
   the bug.
2. **One thread, one loop.** No thread pool underneath. `async` gives you concurrency over I/O,
   never parallelism over CPU. A tight numeric loop inside a coroutine starves every other task.
   `Task.Run` for CPU work maps to `run_in_executor` / `ProcessPoolExecutor` — and in this
   course, usually to "vectorise it in NumPy instead" (P0.2).
3. **No `ConfigureAwait(false)`**, no synchronisation context, and therefore none of the classic
   `.Result` deadlocks. The price: you cannot block on a coroutine from sync code at all.
   `asyncio.run()` is the only door in, and it refuses to nest.
4. **`Task.WhenAll` → `asyncio.gather`**; `WhenAny` → `asyncio.wait(..., FIRST_COMPLETED)`.
   Step 5 measures the 3× on three 100 ms quotes, which is the entire point of the machinery.
5. **`CancellationToken` → cancellation as an exception.** `task.cancel()` raises
   `asyncio.CancelledError` *at the next await point* inside the task. There is no token to
   pass around and no polling; there is also no way to cancel code that never awaits.

## 9. pytest after xUnit

- `[Fact]` → a function named `test_*`. No class, no attribute, no base type.
- `[Theory]` + `[InlineData]` → `@pytest.mark.parametrize`, which is strictly more flexible
  (multiple parameters, ids, stacking).
- `IClassFixture<T>` / ctor injection → **fixtures resolved by argument name**. Declaring
  `def test_x(book)` is the injection; scope (`function`/`module`/`session`) replaces the
  fixture-interface zoo.
- `Assert.Throws<T>` → `pytest.raises(T, match="regex")`. The `match` is worth using: it catches
  "right exception type, wrong reason".
- `monkeypatch` is the built-in for environment and attribute patching — note that it cannot
  poke a field on a **frozen** dataclass, which is why `test_online_paths_refuse_without_opt_in`
  swaps the whole `CONFIG` object instead.

## Gotchas checklist (the things to re-read before Phase 1)

- [ ] `is None`, never `or`, for optional arguments.
- [ ] `field(default_factory=...)` for every mutable default.
- [ ] Return `NotImplemented` from binary dunders you cannot handle.
- [ ] `Decimal` for money; check `bool` before `int`; `ROUND_HALF_UP`, not `round()`.
- [ ] `itertools.groupby` needs sorted input.
- [ ] A generator is single-pass.
- [ ] A coroutine you never await never runs.
- [ ] `async` is not parallelism.
- [ ] `frozen=True` is shallow.

## What breaks if we skip this

Phase 1 builds an autograd engine whose entire design is operator overloading (`__add__`,
`__mul__`, `__pow__`) plus a topological sort over a graph built by those dunders. Without §4 and
§5 above, `Value.__add__` returning `NotImplemented` and a `grad` of `0.0` being falsy will read
as "autograd is confusing" rather than "Python is doing exactly what it said it would".
