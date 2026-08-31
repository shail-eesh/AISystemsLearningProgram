#!/usr/bin/env python3
"""T4 verification benchmark — the capsule's "done when" turned into numbers.

Run:  python3 bench/verify.py            (~7 min on 2 CPU cores)
      python3 bench/verify.py --quick    (~1 min, smaller everything)

The capsule says T4 is done when:

1. the loss curve matches a nanoGPT reference config within noise;
2. generated text is coherent;
3. attention maps show interpretable structure (induction heads found).

Each becomes a measurement with a pass/fail threshold, written to
``bench/results.json``. (1) is split into the stronger claim it implies —
*logit* parity against an independently written reference — plus the curve
comparison against the run-to-run noise band, because "within noise" is only
meaningful once you have measured the noise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parents[3]))

from reference import RefGPT, copy_weights_from, init_like_gpt2  # noqa: E402
from t4_transformer import (  # noqa: E402
    GPT,
    GPTConfig,
    TrainConfig,
    char_dataset,
    get_batch,
    induction_scores,
    smoothed,
    train,
    train_induction_model,
)

LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} [A-Z]+ O \d+\.\d{2} H \d+\.\d{2} L \d+\.\d{2} C \d+\.\d{2} V \d+$"
)


def _cfg(vocab: int, quick: bool) -> GPTConfig:
    return GPTConfig(vocab_size=vocab, block_size=32 if quick else 64,
                     n_layer=2 if quick else 4, n_head=4, n_embd=64 if quick else 128,
                     position="learned", norm="layernorm", dropout=0.0)


# -- 1a. logit parity ------------------------------------------------------


def check_logit_parity(vocab: int, quick: bool) -> dict:
    cfg = _cfg(vocab, quick)
    torch.manual_seed(0)
    ours = GPT(cfg).eval()
    ref = RefGPT(cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd).eval()
    copy_weights_from(ours, ref)
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size - 1))
    with torch.no_grad():
        a, la = ours(idx[:, :-1], idx[:, 1:])
        b, lb = ref(idx[:, :-1], idx[:, 1:])
    scale = float(a.abs().max())
    return {
        "max_abs_logit_diff": float((a - b).abs().max()),
        "relative_logit_diff": float((a - b).abs().max()) / scale,
        "loss_ours": float(la),
        "loss_reference": float(lb),
        "abs_loss_diff": abs(float(la) - float(lb)),
        "tolerance": 1e-5,
        "passed": float((a - b).abs().max()) / scale < 1e-5,
    }


# -- 1b. loss curve vs the reference, against the seed-noise band ----------


def _train_paired(model, data, steps: int, seed: int, block: int, batch: int,
                  lr: float) -> list[float]:
    """Train any module that returns (logits, loss) on a *fixed* batch stream."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(steps):
        x, y = get_batch(data, batch, block, generator=gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def check_loss_curve(data, vocab: int, quick: bool) -> dict:
    """Do the two implementations *learn* the same, not just compute the same?

    Same architecture, same initialisation scheme, same batch stream, three
    seeds each. The comparison is against the seed-to-seed noise band, because
    "matches within noise" is meaningless until the noise has been measured.
    """
    cfg = _cfg(vocab, quick)
    steps = 150 if quick else 400
    batch, lr = 16, 3e-3

    def build_ref(seed: int, gpt2_init: bool):
        torch.manual_seed(seed)
        m = RefGPT(cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd)
        return init_like_gpt2(m, cfg.n_layer) if gpt2_init else m

    def ours_curve(seed: int) -> list[float]:
        torch.manual_seed(seed)
        return _train_paired(GPT(cfg), data, steps, 1234, cfg.block_size, batch, lr)

    def ref_curve(seed: int, gpt2_init: bool = True) -> list[float]:
        return _train_paired(build_ref(seed, gpt2_init), data, steps, 1234,
                             cfg.block_size, batch, lr)

    seeds = [0, 1, 2]
    ours = [smoothed(ours_curve(s), 25) for s in seeds]
    ref = [smoothed(ref_curve(s), 25) for s in seeds]
    default_init = smoothed(ref_curve(0, gpt2_init=False), 25)

    tail = slice(steps // 4, None)   # ignore the warm-up transient

    def spread(curves):
        """Widest seed-to-seed disagreement at any step in the tail."""
        per_step = list(zip(*curves, strict=True))[tail]
        return max(max(vals) - min(vals) for vals in per_step)

    noise = max(spread(ours), spread(ref))
    mean_ours = [sum(c) / len(c) for c in zip(*ours, strict=True)]
    mean_ref = [sum(c) / len(c) for c in zip(*ref, strict=True)]
    gap = max(abs(a - b) for a, b in zip(mean_ours[tail], mean_ref[tail], strict=True))
    return {
        "steps": steps,
        "seeds": seeds,
        "final_ours": mean_ours[-1],
        "final_reference": mean_ref[-1],
        "final_reference_pytorch_default_init": default_init[-1],
        "init_ablation_penalty": default_init[-1] - mean_ref[-1],
        "max_curve_gap": gap,
        "seed_noise_band": noise,
        "gap_over_noise": gap / noise if noise else float("inf"),
        "curve_ours": [round(v, 4) for v in mean_ours[::10]],
        "curve_reference": [round(v, 4) for v in mean_ref[::10]],
        "passed": gap <= noise,
    }


# -- 2. generation coherence ----------------------------------------------


def check_generation(data, val, vocab_obj, quick: bool) -> dict:
    cfg = _cfg(vocab_obj.size, quick).scaled(position="rope")
    torch.manual_seed(0)
    model = GPT(cfg)
    hist = train(model, data, val, TrainConfig(steps=200 if quick else 800,
                                               batch_size=16, lr=3e-3,
                                               eval_every=100, seed=0))
    ids = torch.tensor([vocab_obj.encode("2024-05-14 ")])
    g = torch.Generator().manual_seed(7)
    text = vocab_obj.decode(model.generate(ids, 200 if quick else 800, temperature=0.8,
                                           top_k=20, generator=g)[0])
    lines = [ln for ln in text.split("\n")[1:-1] if ln.strip()]
    wf = sum(bool(LINE.match(ln)) for ln in lines) / max(len(lines), 1)
    distinct = len(set(lines)) / max(len(lines), 1)
    ohlc_ok = 0
    for ln in lines:
        if not LINE.match(ln):
            continue
        p = ln.split()
        o, h, low, c = float(p[3]), float(p[5]), float(p[7]), float(p[9])
        ohlc_ok += int(h >= max(o, c) and low <= min(o, c))
    return {
        "val_loss": hist.final_val,
        "bits_per_char": hist.final_val / 0.6931471805599453,
        "lines_generated": len(lines),
        "well_formed_fraction": wf,
        "distinct_fraction": distinct,
        "ohlc_consistent_fraction": ohlc_ok / max(len(lines), 1),
        "sample": "\n".join(lines[:4]),
        "passed": wf >= (0.4 if quick else 0.9) and distinct >= 0.9,
    }


# -- 3. induction heads ----------------------------------------------------


def check_induction(quick: bool) -> dict:
    """Induction heads, with the two controls the claim actually needs.

    The task is ``variable_period_batch``: each row repeats with its own random
    period drawn from 10..14, so a head that always looks a fixed distance back
    cannot solve it and no positional shortcut exists.

    Control 1 — depth. A one-layer model is run identically. The induction
    circuit needs two layers (a previous-token head feeding a matching head),
    so one layer should plateau well short of the two-layer model.

    Control 2 — attention score is not the circuit. Both models are scored on
    attention mass *and* on behaviour, because a single head can put weight on
    the right position by hedging over a handful of offsets while still being
    unable to predict the token. Only the behavioural gap settles it.
    """
    steps = 800 if quick else 3000
    kw = dict(vocab=64, n_embd=64, n_head=4, length=48, min_period=10, max_period=14)
    t0 = time.perf_counter()
    two, trace = train_induction_model(n_layer=2, steps=steps, record_every=250, **kw)
    probe = induction_scores(two, batch=64, seed=123, length=48,
                             min_period=10, max_period=14)
    one, _ = train_induction_model(n_layer=1, steps=steps, record_every=steps, **kw)
    one_probe = induction_scores(one, batch=64, seed=123, length=48,
                                 min_period=10, max_period=14)
    heads = sorted(probe["per_head"], key=lambda d: -d["score"])
    gap = probe["repeat_accuracy"] - one_probe["repeat_accuracy"]
    return {
        "task": "variable period 10-14, length 48, vocab 64 (no positional shortcut)",
        "chance_level": probe["chance"],
        "best_head": heads[0],
        "top_heads": heads[:4],
        "two_layer_repeat_accuracy": probe["repeat_accuracy"],
        "one_layer_repeat_accuracy": one_probe["repeat_accuracy"],
        "one_layer_best_head_score": one_probe["best"]["score"],
        "depth_accuracy_gap": gap,
        "emergence_trace": trace,
        "steps": steps,
        "seconds": time.perf_counter() - t0,
        "passed": (heads[0]["over_chance"] > 5.0
                   and probe["repeat_accuracy"] > (0.45 if quick else 0.85)
                   and (quick or gap > 0.15)),
    }


# -- 4. KV cache -----------------------------------------------------------


def check_kv_cache(vocab_obj, quick: bool) -> dict:
    cfg = _cfg(vocab_obj.size, quick).scaled(position="rope", block_size=256)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    ids = torch.tensor([vocab_obj.encode("2024-05-14 ")])
    n = 60 if quick else 180
    a = model.generate(ids, n, greedy=True, use_cache=False)
    b = model.generate(ids, n, greedy=True, use_cache=True)
    times = {}
    for use_cache in (False, True):
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            model.generate(ids, n, greedy=True, use_cache=use_cache)
            runs.append(time.perf_counter() - t0)
        times[use_cache] = sorted(runs)[1]
    return {
        "tokens": n,
        "identical": bool(torch.equal(a, b)),
        "seconds_no_cache": times[False],
        "seconds_cached": times[True],
        "speedup": times[False] / times[True],
        "cache_bytes": model.new_caches(1)[0].bytes * cfg.n_layer,
        "passed": bool(torch.equal(a, b)) and times[False] / times[True] > 1.5,
    }


# -- 5. position bake-off --------------------------------------------------


def check_positions(data, val, vocab: int, quick: bool) -> dict:
    steps = 150 if quick else 500
    out = {}
    for pos in ("none", "sinusoidal", "learned", "rope"):
        torch.manual_seed(0)
        m = GPT(_cfg(vocab, quick).scaled(position=pos))
        h = train(m, data, val, TrainConfig(steps=steps, batch_size=16, lr=3e-3,
                                            eval_every=steps, seed=0))
        out[pos] = round(h.final_val, 4)
    best = min(out, key=out.get)
    return {"val_loss": out, "best": best,
            "rope_beats_none": out["rope"] < out["none"], "passed": out["rope"] < out["none"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: fewer steps, looser thresholds, results.json untouched")
    args = ap.parse_args()
    quick = args.quick

    t0 = time.perf_counter()
    train_ids, val_ids, vocab = char_dataset()
    results = {
        "topic": "T4",
        "title": "Transformer from scratch (Attention Is All You Need)",
        "quick": quick,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "corpus": {"chars": len(train_ids) + len(val_ids), "vocab": vocab.size},
    }
    print("1a. logit parity against an independently written reference ...")
    results["logit_parity"] = check_logit_parity(vocab.size, quick)
    print(f"    max relative diff {results['logit_parity']['relative_logit_diff']:.2e}")

    print("1b. loss curve vs the reference, against the seed-noise band ...")
    results["loss_curve"] = check_loss_curve(train_ids, vocab.size, quick)
    lc = results["loss_curve"]
    print(f"    gap {lc['max_curve_gap']:.4f} vs noise band {lc['seed_noise_band']:.4f} "
          f"({lc['gap_over_noise']:.2f}x)")

    print("2.  generation coherence ...")
    results["generation"] = check_generation(train_ids, val_ids, vocab, quick)
    print(f"    val {results['generation']['val_loss']:.4f}, "
          f"well-formed {results['generation']['well_formed_fraction']:.0%}")

    print("3.  induction heads ...")
    results["induction"] = check_induction(quick)
    ind = results["induction"]
    print(f"    best head L{ind['best_head']['layer']}H{ind['best_head']['head']} "
          f"scores {ind['best_head']['score']:.3f} "
          f"({ind['best_head']['over_chance']:.1f}x chance); "
          f"2-layer copies {ind['two_layer_repeat_accuracy']:.0%} vs "
          f"1-layer {ind['one_layer_repeat_accuracy']:.0%}")

    print("4.  KV cache ...")
    results["kv_cache"] = check_kv_cache(vocab, quick)
    print(f"    identical={results['kv_cache']['identical']}, "
          f"speedup {results['kv_cache']['speedup']:.2f}x")

    print("5.  position bake-off ...")
    results["positions"] = check_positions(train_ids, val_ids, vocab.size, quick)
    print(f"    {results['positions']['val_loss']}")

    results["seconds"] = time.perf_counter() - t0
    checks = {k: v["passed"] for k, v in results.items()
              if isinstance(v, dict) and "passed" in v}
    results["checks"] = checks
    results["all_passed"] = all(checks.values())
    if quick:
        print("\n  --quick is a smoke test: everything is trained for a fraction of the"
              "\n  steps, thresholds are loosened, and results.json is NOT overwritten."
              "\n  The committed numbers always come from a full run.")
    else:
        (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\n  {sum(checks.values())}/{len(checks)} checks passed in "
          f"{results['seconds']:.0f}s" + ("" if quick else " -> bench/results.json"))
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
