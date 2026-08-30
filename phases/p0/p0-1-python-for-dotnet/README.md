# P0.1 · Python for the .NET veteran

**Phase:** Python & Tensor Ramp (P0) · **Generation day:** Day 1 · **Video episodes:** 1
· **Status:** ✅ code · ✅ tests · ✅ bench

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

The AlphaDesk **paper OMS domain model** — `Money`, `Quantity`, `Order`, `Fill`, `Position`,
`Portfolio`, plus pre-trade risk checks — ported from the C# idioms you already know into
idiomatic Python, with the tests that prove it.

This is not a "learn Python syntax" topic. It is a **translation layer**: every construct you
reach for in .NET has a Python counterpart that is *almost* the same, and the gap is where the
bugs live. The domain is deliberately one you can judge on sight, so your attention stays on the
language rather than on the problem.

## Why this first

Everything after Phase 0 is numerical code where the language is invisible. If dataclasses,
dunder methods, protocols, context managers and `asyncio` are still novel then, you will
misread every subsequent topic as "PyTorch weirdness" when it is really Python semantics.

## The .NET → Python map this topic covers

| C# / .NET | Python | The gap that bites |
|---|---|---|
| POCO / `record` | `@dataclass` | `frozen=True` is shallow; mutable defaults need `field(default_factory=...)` |
| ctor guard clauses | `__post_init__` | runs *after* field assignment, not before |
| `decimal` | `decimal.Decimal` | `float` is as wrong here as `double` is there — and `bool` is an `int` |
| `enum` + extension methods | `Enum` with methods/properties | `str`-backed enums parse from the wire and compare equal to their string |
| `interface` | `typing.Protocol` | structural, not nominal — nothing is declared or implemented |
| operator overloading | `__add__`, `__eq__`, `__lt__` | return `NotImplemented` (not `NotImplementedError`) for unknown operands |
| `ToString()` | `__str__` / `__repr__` | two of them: one for humans, one for debugging |
| `IDisposable` + `using` | `__enter__`/`__exit__` + `with` | `__exit__` sees the exception and can swallow it |
| `IEnumerable<T>` + `yield return` | generator function | single-pass; `itertools` is the LINQ standard library |
| `Task` + `async/await` | coroutine + `async/await` | coroutines are **cold**; one thread, one loop; no `ConfigureAwait` |
| xUnit `[Fact]`/`[Theory]` | `pytest` + `parametrize` | fixtures by argument name, no attributes, no base class |
| NuGet / `.csproj` | `uv` / `pyproject.toml` | no assemblies; a package is a directory with `__init__.py` |

## How to run

```bash
# from the repo root
python3 -m pytest phases/p0/p0-1-python-for-dotnet -q        # 63 tests
python3 phases/p0/p0-1-python-for-dotnet/bench/replay_orders.py

# the step ladder — each script is standalone and prints its own walkthrough
cd phases/p0/p0-1-python-for-dotnet
python3 steps/step1_dataclasses_vs_pocos.py
python3 steps/step2_enums_protocols.py
python3 steps/step3_dunder_operators.py
python3 steps/step4_context_and_generators.py
python3 steps/step5_async_vs_tasks.py
```

## The step ladder

1. **`step1_dataclasses_vs_pocos.py`** — value semantics, shallow immutability, the mutable-default
   trap, guard clauses, and free introspection via `fields()`.
2. **`step2_enums_protocols.py`** — enums that carry behaviour (`Side.BUY.sign`), and a risk check
   that satisfies `RiskCheck` without ever naming it.
3. **`step3_dunder_operators.py`** — `Money` arithmetic, why `float` is refused, the
   `NotImplemented` protocol, ordering, and the truthiness trap.
4. **`step4_context_and_generators.py`** — an all-or-nothing order batch as a context manager;
   a fill stream as a lazy generator; `itertools` as LINQ.
5. **`step5_async_vs_tasks.py`** — cold coroutines, `gather` vs `Task.WhenAll`, cancellation,
   `asyncio.timeout`, async generators, and why `async` never parallelises CPU work.

## Verification benchmark ("done when")

`bench/replay_orders.py` replays all 210 orders of the synthetic OMS tape
(`common/data/samples/orders_sample.csv`) through the domain model and reconciles the result
**twice**, against two references that share no code with it:

* net quantity per symbol vs a pandas `groupby` of signed filled shares — **0 breaks**;
* realised P&L per symbol vs an independently written float average-cost reducer —
  worst relative error **3.1e-07**, inside the 1e-6 tolerance.

Committed output: [`bench/results.json`](bench/results.json). The reconciliation runs inside
pytest too (`tests/test_bench_reconciliation.py`), so the claim cannot rot.

> The residual is *float noise in the reference*, not error in the model. Chasing it is how the
> `Money` storage precision ended up at six decimal places rather than four — see NOTES.

## AlphaDesk hook

`src/p0_1_oms/alphadesk_hook.py` registers two components on the desk:

| key | surface | what it gives AlphaDesk |
|---|---|---|
| `orders.paper_book` | Order Workflow | the `Portfolio` every later paper order lands in |
| `compliance.default_risk_checks` | Compliance | max-notional + tradable-universe pre-trade checks |

From Phase 4 the ReAct agent will build an order ticket and place it into exactly this book.

## Layout

- `src/p0_1_oms/` — the implementation (`money.py`, `oms.py`, `errors.py`, `alphadesk_hook.py`)
- `steps/` — the step ladder (ordered, individually runnable)
- `tests/` — pytest (63 tests; must pass for the topic to be "done")
- `bench/` — the reconciliation benchmark + committed `results.json`
- `NOTES.md` — intuition + gotchas (mirrors the video script)

## Videos

Episode script: [`video/topics/p0.1/script.md`](../../../video/topics/p0.1/script.md).
Rendered `.mp4`s are delivered in chat, not committed.

---
*AlphaDesk is a fictional educational simulation — no real orders, money, brokerage systems, or market-data redistribution.*
