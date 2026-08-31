#!/usr/bin/env bash
# One command that says whether the repo is green.
#
#   bash scripts/verify.sh          # tests + lint + benchmarks + video scripts
#   bash scripts/verify.sh --fast   # skip the benchmarks
#
# A daily generation run must leave this passing. If it does not, the run
# should have stopped and checkpointed instead.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FAST="${1:-}"
FAILED=()

step() {
  local name="$1"; shift
  printf '\n\033[1m== %s\033[0m\n' "$name"
  if "$@"; then
    printf '\033[32m   ok\033[0m\n'
  else
    printf '\033[31m   FAILED\033[0m\n'
    FAILED+=("$name")
  fi
}

step "pytest"        python3 -m pytest -q
step "ruff"          python3 -m ruff check .
step "index/ledger"  python3 scripts/gen_index.py
step "video scripts" python3 video/kokoro/narrate.py --check --all
step "scene props"    python3 scripts/check_scene_props.py

if [ "$FAST" != "--fast" ]; then
  step "P0.1 bench" python3 phases/p0/p0-1-python-for-dotnet/bench/replay_orders.py
  step "P0.2 bench" python3 phases/p0/p0-2-numpy-as-linq/bench/indicator_parity.py
  if python3 -c "import torch" 2>/dev/null; then
    step "P0.3 bench" python3 phases/p0/p0-3-pytorch-training-loop/bench/train_and_report.py
  else
    printf '\n\033[1m== P0.3 bench\033[0m\n   skipped (optional [torch] extra not installed)\n'
  fi
  step "T31 bench"  python3 phases/p1/t31-autograd/bench/gradcheck_suite.py
  step "T16A bench" python3 phases/p1/t16a-matmul-cpu/bench/roofline.py
  step "T45A bench" python3 phases/p1/t45a-softmax-cpu/bench/numerics.py
  step "T30 bench"  python3 phases/p1/t30-tokenizer-bpe/bench/compression.py
fi

# The index generator rewrites LEDGER.md and index.html; they must already match
# status.json, or the committed hub is stale.
if ! git diff --quiet -- index.html EXECUTION/LEDGER.md 2>/dev/null; then
  printf '\n\033[31m== index.html / LEDGER.md are stale — commit the regenerated files\033[0m\n'
  FAILED+=("index freshness")
fi

printf '\n'
if [ ${#FAILED[@]} -eq 0 ]; then
  printf '\033[32mall checks passed\033[0m\n'
  exit 0
fi
printf '\033[31mfailed: %s\033[0m\n' "${FAILED[*]}"
exit 1
