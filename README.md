# AI Systems Forge — AI Systems Learning Program

A build-it-yourself mastery program: **51 "Build Your Own X" topics** across
**10 phases**, each with slow-paced Remotion video lessons (narrated with Kokoro-82M),
step-laddered practice code with tests, and one capital-markets capstone — **AlphaDesk**.

## Start here
Open **[`index.html`](index.html)** — the navigation hub. It links every phase, every topic,
each topic's videos, and live generation progress.

## How this repo is built
Content is generated incrementally by a scheduled Cowork task (a fresh Claude Opus session
each day). It follows [`EXECUTION/DAILY_PROMPT.md`](EXECUTION/DAILY_PROMPT.md), reads
[`EXECUTION/LEDGER.md`](EXECUTION/LEDGER.md) to find the next work, builds it, and pushes here.
See [`MASTER_PLAN.md`](MASTER_PLAN.md) for the full plan and the day-by-day schedule.

## Navigate
- **[MASTER_PLAN.md](MASTER_PLAN.md)** — full curriculum, per-topic capsules, capstone architecture
- **[EXECUTION/DAY_PLAN.md](EXECUTION/DAY_PLAN.md)** — the 15-day generation schedule
- **[EXECUTION/LEDGER.md](EXECUTION/LEDGER.md)** — per-topic progress (code / tests / video / bench)
- **[phases/](phases/)** — every topic in its own folder (identical shape)
- **[video/](video/)** — the reusable component library + per-topic scripts
- **[gpu-runner/](gpu-runner/)** — scripts you run on your RTX 4070

## Verify it locally

```bash
pip install -e ".[dev,torch]"    # torch is optional; P0.3 skips without it
bash scripts/verify.sh           # tests + lint + benchmarks + video scripts
```

Every generation run must leave this passing.

## Repo status
Total topics: **51** (52 build modules) · Total planned video episodes: **168** ·
Branch: `main`

**Phase 0 complete** (Day 1): 232 tests passing, `ruff` clean, three verification benchmarks,
three narrated video episodes.

**Phase 1 complete** (Day 2): T31 autograd · T16A matmul-CPU · T45A softmax-CPU · T30 tokenizer/FinTok.
**261 tests passing**, `ruff` clean, four verification benchmarks, ten narrated episodes
(~1h34m). Headlines: the hand-rolled engine matches an independently derived reference loss
curve to **6e-17**; cache blocking alone beats naive C by **22.6x**; online softmax is
**bit-identical** to two-pass in f64/f32/f16; FinTok round-trips **100,000** fuzzed strings and
compresses financial text **2.73x** better than a matched general-domain vocabulary.

See [`EXECUTION/runs/`](EXECUTION/runs/) for the per-run log, including the two deviations from
the plan this run made and why.

---
*AlphaDesk is a fictional educational simulation — no real orders, money, brokerage systems, or market-data redistribution.*
