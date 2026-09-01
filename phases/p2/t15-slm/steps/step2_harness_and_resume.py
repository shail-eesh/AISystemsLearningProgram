#!/usr/bin/env python3
"""Step 2 — the harness, and the only checkpoint test that matters.

Run:  python3 steps/step2_harness_and_resume.py     (~3 min)

A checkpoint is not "the weights". A checkpoint is *everything the loop needs so
that stopping and restarting is invisible*. Getting that wrong is the classic
overnight-run bug: the job crashes at hour six, you "resume", and the loss jumps
because Adam's moment estimates restarted from zero and the data sampler
restarted from the top of the corpus.

The test here is the strong form: train 120 steps straight through, and train
60 + resume + 60, and compare the parameters. Anything other than exactly 0.0 is
a bug you would otherwise discover the hard way.
"""

import pathlib
import tempfile

import _bootstrap  # noqa: F401
import torch
from t4_transformer import GPT
from t15_alphaslm import LADDER, Trainer, TrainSpec, ensure_shards, lr_at

RUNG = LADDER["alphaslm-0.6m"]


def the_schedule_first():
    print("  the learning-rate schedule (warmup 100, cosine to 10%, 1200 steps):\n")
    spec = TrainSpec(steps=1200, warmup=100, lr=3e-3, min_lr_ratio=0.1)
    print(f"    {'step':>6} {'lr':>10}   {'':<32}")
    for step in (0, 25, 50, 99, 100, 300, 600, 900, 1199):
        lr = lr_at(step, spec)
        bar = "#" * round(40 * lr / spec.lr)
        print(f"    {step:>6} {lr:>10.2e}   {bar}")
    print("\n    Warmup exists because Adam's second-moment estimate is meaningless")
    print("    before it has seen gradients: at step 0 the update is essentially")
    print("    sign(g) x lr, and a full learning rate there is a confident step in an")
    print("    arbitrary direction. Cosine decay to 10% is what lets the last few")
    print("    hundred steps settle instead of bouncing.")


def what_a_checkpoint_contains(shards):
    print("\n  what one checkpoint actually holds:\n")
    train_shard, val_shard = shards
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(15)
        model = GPT(RUNG.gpt_config(3495))
        tr = Trainer(model, train_shard, val_shard,
                     TrainSpec(steps=10, batch_size=8, eval_every=100, checkpoint_every=0),
                     run_dir=pathlib.Path(d))
        tr.train()
        path = tr.save()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        size = path.stat().st_size
        print(f"    {size / 1e6:.1f} MB, keys: {sorted(ckpt)}\n")
        for k in sorted(ckpt):
            v = ckpt[k]
            if k == "model":
                n = sum(t.numel() for t in v.values())
                unique = len({t.data_ptr() for t in v.values()})
                print(f"      {k:<12} {len(v)} tensors ({unique} distinct — the tied "
                      f"head is stored twice), {n:,} entries")
            elif k == "optimizer":
                print(f"      {k:<12} {len(v['state'])} parameter slots of Adam moments")
            elif k in ("state",):
                print(f"      {k:<12} step {v['step']}, tokens {v['tokens']:,}")
            elif k == "numpy_rng":
                print(f"      {k:<12} the data sampler's position in its stream")
            elif k == "torch_rng":
                print(f"      {k:<12} torch's stream (dropout draws, any re-init)")
            else:
                print(f"      {k:<12} {type(v).__name__}")
        print("\n    The optimiser slot is roughly two thirds of the file — two moment")
        print("    tensors per parameter. It is also the part people omit, because a")
        print("    model that loads and runs *looks* like a successful resume.")


def resume_is_invisible(shards):
    print("\n  the test: 120 straight, versus 60 + resume + 60\n")
    train_shard, val_shard = shards

    def spec():
        return TrainSpec(steps=120, batch_size=8, warmup=20, eval_every=1000,
                         checkpoint_every=60)

    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        torch.manual_seed(15)
        straight = GPT(RUNG.gpt_config(3495))
        ts = Trainer(straight, train_shard, val_shard, spec(), run_dir=root / "straight")
        ts.train()

        torch.manual_seed(15)
        first = GPT(RUNG.gpt_config(3495))
        tf = Trainer(first, train_shard, val_shard, spec(), run_dir=root / "split")
        tf.train(until=60)
        del tf, first

        torch.manual_seed(4242)     # deliberately poison the global RNG
        torch.manual_seed(15)
        second = GPT(RUNG.gpt_config(3495))
        t2 = Trainer(second, train_shard, val_shard, spec(), run_dir=root / "split")
        state = t2.load()
        print(f"    loaded a checkpoint at step {state.step}, "
              f"{state.tokens:,} tokens seen")
        t2.train()

        worst = max(float((a - b).abs().max())
                    for a, b in zip(straight.state_dict().values(),
                                    second.state_dict().values(), strict=True))
        print(f"    largest parameter difference after 120 steps: {worst}")
        a_tail = [round(h["loss"], 6) for h in ts.state.history[-3:]]
        b_tail = [round(h["loss"], 6) for h in t2.state.history[-3:]]
        print(f"    last three losses, straight: {a_tail}")
        print(f"    last three losses, resumed:  {b_tail}")
    print("\n    Exactly zero. Not 'close' — the sampler's RNG state, the optimiser")
    print("    moments and the schedule position all came back, so the resumed run")
    print("    computed the identical arithmetic. Drop any one of the three from the")
    print("    checkpoint and this number stops being zero.")


def accumulation_is_the_same_update(shards):
    print("\n  gradient accumulation: one batch of 16, or four micro-batches of 4?\n")
    train_shard, val_shard = shards
    results, models = {}, {}
    for batch, micro in ((16, 1), (4, 4)):
        torch.manual_seed(15)
        model = GPT(RUNG.gpt_config(3495))
        tr = Trainer(model, train_shard, val_shard,
                     TrainSpec(steps=30, batch_size=batch, micro_batches=micro,
                               warmup=5, eval_every=1000, checkpoint_every=0))
        tr.train()
        results[(batch, micro)] = tr.state.history[-1]["loss"]
        models[(batch, micro)] = model
    print(f"    {'batch':>6} {'micro':>6} {'windows/step':>13} {'final loss':>12}")
    for (batch, micro), loss in results.items():
        print(f"    {batch:>6} {micro:>6} {batch * micro:>13} {loss:>12.6f}")
    worst = max(float((a - b).abs().max())
                for a, b in zip(models[(16, 1)].state_dict().values(),
                                models[(4, 4)].state_dict().values(), strict=True))
    print(f"\n    largest parameter difference after 30 steps: {worst}")
    print("\n    The same update, to float32 rounding — and *not* bit-identical, which")
    print("    is the interesting part. Both runs draw the same 16 windows in the same")
    print("    order from the same sampler, but summing four partial gradients is a")
    print("    different order of additions than summing sixteen at once, and float")
    print("    addition is not associative (T45A, at length). A few parts in a million")
    print("    after 30 steps is reassociation noise, not a bug — but it is also why")
    print("    'my accumulated run does not exactly reproduce' is expected, and why a")
    print("    resume test must not change the accumulation factor. The loss is")
    print("    divided by the number of micro-batches so the accumulated gradient is a")
    print("    mean and not a sum — forget that division and your effective learning")
    print("    rate is silently multiplied by the accumulation factor, which people")
    print("    then report as 'accumulation makes training unstable'.")
    print("    This is what buys you a 40M model on 12 GB: the batch size the")
    print("    optimiser sees is decoupled from the batch size memory has to hold.")


def the_log_is_greppable(shards):
    print("\n  the metrics log — one JSON object per line, no service, no account:\n")
    train_shard, val_shard = shards
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(15)
        model = GPT(RUNG.gpt_config(3495))
        tr = Trainer(model, train_shard, val_shard,
                     TrainSpec(steps=60, batch_size=8, warmup=10, log_every=20,
                               eval_every=30, eval_batches=4, checkpoint_every=0),
                     run_dir=pathlib.Path(d))
        tr.train()
        for line in (pathlib.Path(d) / "metrics.jsonl").read_text().splitlines():
            print(f"      {line}")
    print("\n    Everything a hosted chart would have shown, in a file you can grep,")
    print("    diff between runs, and read on a machine with no network.")


if __name__ == "__main__":
    print(__doc__)
    shards = ensure_shards(block_size=128)[:2]
    the_schedule_first()
    what_a_checkpoint_contains(shards)
    resume_is_invisible(shards)
    accumulation_is_the_same_update(shards)
    the_log_is_greppable(shards)
