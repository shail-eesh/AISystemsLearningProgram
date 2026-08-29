# Daily Generation Run — Operating Instructions

You are a fresh Claude **Opus** Cowork session. Your job: advance the AI Systems Forge course by one
work package, commit it, and hand the human the day's bundle + videos. Work carefully and finish clean.
These instructions are the contract; follow them exactly.

> **Delivery model (important):** this cloud environment's git proxy can *read* this public repo but
> **cannot push** to it. So you CLONE to get the latest state, build on top, commit locally, and deliver
> the day's work as a **git bundle** for the human to push. No tokens are used anywhere.

## 0. Setup
```bash
cd /home/claude && rm -rf repo
git clone https://github.com/shail-eesh/AISystemsLearningProgram.git repo   # public read via the proxy
cd repo && git checkout course-generation
git config user.email "cowork@local"; git config user.name "AI Systems Forge (Opus)"
```
**Setup self-check:** if the clone fails, or the repo has no `course-generation` branch / no
`MASTER_PLAN.md`, the one-time setup isn't done yet. Do NOT start building — post a short message asking
the human to (a) make the repo public and (b) push the scaffold bundle to `course-generation`, then stop.
Otherwise read, in order: `MASTER_PLAN.md` (source of truth — the capsule for each topic you'll build),
`EXECUTION/DAY_PLAN.md` (the schedule), `EXECUTION/LEDGER.md` (current status).

## 1. Choose the work
Scan `LEDGER.md` top to bottom. Your target set is:
1. **Catch-up first:** every earlier row whose **Code** is not ✅ (spillover from prior runs).
2. **Then today's package:** the rows for the current generation day. Determine the current day as
   `1 + (number of days whose rows are already fully code-complete)`, i.e. the first day that still
   has unfinished rows. On the very first run, that's Day 1 (infra).

Do **Day 1 infra** (below) before any topic, if not already ✅ in the ledger's infra checklist.

## 2. Build each topic (code-first, then video)
For every topic in your target set, in ledger order, follow its capsule in `MASTER_PLAN.md` and the
per-topic folder contract:

**A. Code (must-have).**
- Write `phases/<folder>/README.md` (fill the stub: what/why, the paper, how to run) and `NOTES.md`
  (intuition + gotchas — this doubles as the video script source).
- Implement the **step ladder** from the capsule under `steps/` as ordered, individually-runnable
  checkpoints, and the consolidated implementation under `src/`.
- Write `tests/` (pytest, or `cargo test` for Rust). **A topic's code is done only when tests pass.**
- Run the **verification benchmark** from the capsule; write results to `bench/`. If it's a GPU topic,
  commit the CUDA/Triton source + a CPU-verified reference (Triton `TRITON_INTERPRET=1` / NumPy), add
  the runner to `gpu-runner/`, and mark **Bench = 🖥️ awaiting-4070**.
- Wire the topic's **AlphaDesk hook** into `common/alphadesk/` (import/registration + a smoke test).
- Keep scale small (minutes-to-hours on CPU/4070). Commit per step: `git add -A && git commit -m "..."`.

**B. Video (best-effort).**
- Write `video/topics/<id>/script.md` following the pacing rules in MASTER_PLAN §5 (one idea per scene,
  ~120–135 wpm, three-way math reveal, line-highlighted code, recap bookends).
- Build `scenes.tsx` by parametrizing the shared components in `video/src/components/`.
- Synthesize narration with the Kokoro-82M pipeline (`video/kokoro/` from Day 1; voice fixed in
  `video/VOICE.md`), producing `audio/` wavs + `timings.json`.
- `remotion render` at 1280×720/30fps to an `out/` mp4 (gitignored).
- **If you are running low on time/tokens, stop after Code for the remaining topics** and mark their
  **Video = 🟡 pending**. Never skip code to do video.

## 3. Day 1 infra (only on the first run, or if the infra checklist isn't all ✅)
- `common/`: config, logging, and market-data loaders — NSE bhavcopy, yfinance OHLCV, SEC EDGAR
  full-text (fair-use excerpts), plus small cached sample datasets committed under `common/data/`.
- `common/alphadesk/`: empty package with a component registry the topics plug into.
- `video/src/components/`: the reusable Remotion library + `theme.ts` (TitleCard, ConceptScene,
  MathReveal, CodeWalkthrough, DiagramScene, ChartScene, ArchitectureMap, RecapScene, Callout,
  Transition) and `video/kokoro/` narration pipeline (ONNX, CPU) + `video/VOICE.md` (pick one fixed
  American-English voice, e.g. an `af_*`/`am_*` voice, and record the settings).
- `scripts/gen_index.py`: regenerates `index.html` from `EXECUTION/LEDGER.md` (statuses + links).
- Then build the Phase 0 topics (P0.1–P0.3).

## 4. Update state, bundle, & deliver
- Update `EXECUTION/status.json` for every topic you touched (code/tests/bench/video/wired + a short
  `note`), then run `python3 scripts/gen_index.py` — it regenerates BOTH `index.html` and
  `EXECUTION/LEDGER.md` from that JSON. Write a full log to `EXECUTION/runs/<YYYY-MM-DD>.md`.
- **Commit locally:** `git add -A && git commit -m "Day N: <summary>"`. (Commit per step as you go.)
- **Bundle the branch for the human to push:**
  ```bash
  git bundle create /home/claude/forge-day-N.bundle course-generation
  ```
  `SendUserFile` that bundle. The human updates GitHub with:
  `git fetch /path/forge-day-N.bundle course-generation && git push origin FETCH_HEAD:course-generation`
  (their local clone fast-forwards because you built on top of the latest public state).
- **Deliver videos:** `SendUserFile` each rendered `.mp4` with a one-line caption (topic + episode).
  The mp4s are gitignored — they reach the human only through this delivery.
- Post a short summary: code-done / tests-passed / video shipped-vs-pending / GPU benches awaiting the
  4070 / **and the exact push command above** so the human can update the repo in one step.

## 5. Budget guard & termination
- If low on budget at any point: finish the current step, commit, update the ledger to reflect exactly
  what's done, push, and stop cleanly. The next run resumes from the ledger.
- **When the whole course is complete** (every topic Code ✅ / Tests ✅ / Bench ✅-or-🖥️ / Wired ✅, and
  the Phase 9 capstone assembled): do a final `index.html` regen and portfolio-video pass, then
  **delete the scheduled task** — list scheduled tasks, find the one named "AI Systems Forge — daily
  generation", and delete it — and post a "course complete" summary.

## Guardrails
- AlphaDesk is a **fictional educational simulation**: no real orders, money, brokerage systems, or
  market-data redistribution. Keep the disclaimers.
- Small scale always. Reference libraries (tiktoken, hnswlib, vLLM, TRL…) appear only as benchmarks,
  never as the thing being learned.
- Leave the repo green: never commit failing tests or a broken build.
