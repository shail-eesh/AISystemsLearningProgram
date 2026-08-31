#!/usr/bin/env python3
"""Step 7 — the KV cache: same tokens, far less arithmetic.

Run:  python3 steps/step7_kv_cache.py     (~2 min)

Generating token t without a cache means running the whole prefix through every
layer again. But the keys and values for positions 0..t-1 cannot have changed —
the mask guarantees nothing in the past depends on the future. Recomputing them
is pure waste, and it is *quadratic* waste: generating n tokens costs O(n^2)
prefix work instead of O(n).

The cache stores k and v per layer as they are produced. Three claims, all
measured below: it is exact, it is faster, and it costs memory that grows
linearly with context — which is the constraint T12's paged cache exists to
manage.
"""

import time

import _bootstrap  # noqa: F401
import torch
from t4_transformer import GPT, GPTConfig, KVCache, TrainConfig, char_dataset, train


def build_model():
    tr, va, vocab = char_dataset()
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=vocab.size, block_size=256, n_layer=4, n_head=4,
                          n_embd=128, position="rope"))
    train(model, tr, va, TrainConfig(steps=300, batch_size=16, lr=3e-3, eval_every=300))
    return model.eval(), vocab


def exactness_first(model, vocab) -> None:
    print("  the cache must not change the answer:\n")
    ids = torch.tensor([vocab.encode("2024-05-14 ")])
    for n in (16, 64, 128):
        a = model.generate(ids, n, greedy=True, use_cache=False)
        b = model.generate(ids, n, greedy=True, use_cache=True)
        print(f"    {n:>4} greedy tokens: identical = {torch.equal(a, b)}")
    print("\n    Greedy, because it removes the RNG from the comparison. Same argmax at")
    print("    every step means the same logits at every step — which is the claim.")


def where_the_savings_are(model, vocab) -> None:
    print("\n  prefix work, counted rather than timed:\n")
    print(f"    {'tokens generated':>17} {'no cache: positions re-run':>28} "
          f"{'cache: positions run':>22} {'ratio':>7}")
    prompt = 11
    for n in (16, 32, 64, 128):
        no_cache = sum(min(prompt + i, model.config.block_size) for i in range(n))
        cached = prompt + n
        print(f"    {n:>17} {no_cache:>28,} {cached:>22,} {no_cache / cached:>7.1f}x")
    print("\n    Every one of those re-runs recomputes keys and values that are")
    print("    bit-for-bit identical to the previous step's. The cache is not an")
    print("    optimisation trick; it is the removal of an obvious mistake.")


def wall_clock(model, vocab) -> None:
    print("\n  wall clock on 2 CPU cores (median of 3):\n")
    ids = torch.tensor([vocab.encode("2024-05-14 ")])
    print(f"    {'tokens':>7} {'no cache (s)':>14} {'cached (s)':>12} {'speedup':>9} "
          f"{'tok/s cached':>14}")
    for n in (32, 64, 128, 200):
        times = {}
        for use_cache in (False, True):
            runs = []
            for _ in range(3):
                t0 = time.perf_counter()
                model.generate(ids, n, greedy=True, use_cache=use_cache)
                runs.append(time.perf_counter() - t0)
            times[use_cache] = sorted(runs)[1]
        print(f"    {n:>7} {times[False]:>14.2f} {times[True]:>12.2f} "
              f"{times[False] / times[True]:>8.2f}x {n / times[True]:>14.1f}")
    print("\n    The speedup grows with length, because the thing being avoided grows")
    print("    with length. At tiny lengths the cache can even lose: it pays fixed")
    print("    per-step Python and allocation cost to save arithmetic there is not")
    print("    much of yet.")


def what_it_costs(model) -> None:
    print("\n  and what the cache costs in memory:\n")
    c = model.config
    print(f"    {'context':>8} {'batch':>6} {'cache bytes':>14} {'per token':>11}")
    for block in (256, 1024, 4096):
        for batch in (1, 8):
            per_layer = KVCache.empty(batch, c.n_head, block, c.head_dim).bytes
            total = per_layer * c.n_layer
            print(f"    {block:>8} {batch:>6} {total:>14,} {total / (block * batch):>11.0f}")
    print("\n    Linear in context x batch x layers x heads x head_dim x 2 x 4 bytes.")
    print("    A megabyte here at 256 slots. For a 7B model at 4k context it is")
    print("    gigabytes *per sequence*, which is why a production server cannot simply")
    print("    preallocate one contiguous block per request and hope — the allocation")
    print("    is bigger than the model's activations and it is mostly empty most of")
    print("    the time. That is the exact problem T12 (paged KV cache) solves.")


def the_wall(model, vocab) -> None:
    print("\n  the naive cache's hard edge:\n")
    ids = torch.tensor([vocab.encode("2024-05-14 ")])
    over = model.config.block_size - ids.shape[1] + 5
    try:
        model.generate(ids, over, greedy=True, use_cache=True)
    except ValueError as exc:
        print(f"    asking for {over} tokens on a {model.config.block_size}-slot cache:")
        print(f"      ValueError: {exc}")
    print("\n    A fixed contiguous cache is a hard context limit, and it fails loudly")
    print("    here on purpose. Sliding it, paging it, or evicting from it are all")
    print("    later topics — but you cannot appreciate those designs until you have")
    print("    hit this wall with your own model.")


if __name__ == "__main__":
    print(__doc__)
    model, vocab = build_model()
    exactness_first(model, vocab)
    where_the_savings_are(model, vocab)
    wall_clock(model, vocab)
    what_it_costs(model)
    the_wall(model, vocab)
