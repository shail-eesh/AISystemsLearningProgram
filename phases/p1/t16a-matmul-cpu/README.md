# T16A · Matrix multiplication kernel — CPU

**Phase:** Foundations (P1) · **Generation day:** Day 2 · **Video episodes:** 2
· *(the GPU half of this topic is T16B, Phase 7)*

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

The same 2·n³ floating point operations, written five ways, each one differing
from the last only in **the order it touches memory**:

1. `matmul_naive_ijk` — the textbook triple loop. The inner loop walks a column
   of B, using 8 bytes of every 64-byte cache line.
2. `matmul_ikj` — two loops swapped. One line of code.
3. `matmul_blocked` — tile so a panel of B stays resident while several rows of
   A consume it.
4. `matmul_blocked_omp` — the column tiles of C are disjoint, so threads need
   no locks.
5. `matmul_blocked_regtile` — four rows of A at a time, so one load of B feeds
   four FMAs. The germ of a real micro-kernel.

Plus the accounting that turns seconds into a defensible number: FLOPs,
arithmetic intensity, and a bandwidth-vs-working-set curve measured on the
machine you are sitting at.

## Results (2-core Xeon @ 2.10 GHz, AVX-512, gcc `-O3 -march=native -fopenmp`)

All five kernels compiled with **identical flags** — handicapping the naive one
would produce a better number and a worse lesson.

| n=1024 | ms | GFLOP/s | vs naive |
|:--|--:|--:|--:|
| naive i-j-k | 4098 | 0.5 | 1× |
| i-k-j (one swapped loop) | 259 | 8.3 | **15.8×** |
| blocked, single thread | 181 | 12.2 | **22.6×** |
| blocked + 2 threads | 125 | 17.2 | 32.8× |
| blocked + register tile | 115 | 18.7 | **35.6×** |
| NumPy / OpenBLAS | 16 | 135.0 | 256× |

The capsule asked for ≥10× over naive C from blocking alone; single-threaded
blocking gives **22.6×**. Against OpenBLAS we reach **14%** of its throughput,
and the honest explanation is in `NOTES.md`: this kernel still writes C to
memory on every k step, where a real micro-kernel accumulates a tile of C in
vector registers across the whole k loop.

**Bandwidth vs working set** (in-place scale, read+write):

| 16 KB | 64 KB | 256 KB | 1 MB | 4 MB | 16 MB | 64 MB |
|--:|--:|--:|--:|--:|--:|--:|
| 41.5 | 54.2 | 70.9 | **81.1** | 44.6 | 43.2 | 33.8 GB/s |

The cliff between 1 MB and 4 MB is a cache level ending, and it is exactly why
the block sweep prefers a B tile of ~256 KB. Read the cliffs, not the absolute
numbers.

## AlphaDesk hook

None, by design — the desk uses NumPy/BLAS like any sane system, and the capsule
says so. What is registered on the **foundation** surface is the literacy
artefact `foundation.cpu_matmul_kernels`: the kernels, their FLOP accounting,
and the CPU baseline that Phase 7's GPU pass (T16B) has to beat.

AlphaDesk is a fictional educational simulation — no real orders, money, or
venues anywhere in this repository.

## How to run

```bash
python3 phases/p1/t16a-matmul-cpu/steps/step1_naive_python.py
python3 phases/p1/t16a-matmul-cpu/steps/step2_numpy_and_flops.py
python3 phases/p1/t16a-matmul-cpu/steps/step3_naive_c.py
python3 phases/p1/t16a-matmul-cpu/steps/step4_loop_order_and_blocking.py
python3 phases/p1/t16a-matmul-cpu/steps/step5_simd_and_threads.py
python3 phases/p1/t16a-matmul-cpu/steps/step6_roofline_table.py

python3 -m pytest phases/p1/t16a-matmul-cpu/tests -q
python3 phases/p1/t16a-matmul-cpu/bench/roofline.py
```

The shared library is compiled on first import into a gitignored `build/`
directory (`cc -O3 -march=native -fopenmp -shared -fPIC`). With no compiler on
`PATH` the module reports `AVAILABLE = False` and the tests skip rather than
fail — the same degrade-don't-die rule the AlphaDesk registry follows.

## Layout

- `src/t16a_matmul/kernels.c` — the five kernels, heavily commented
- `src/t16a_matmul/native.py` — on-demand compile + ctypes binding
- `src/t16a_matmul/reference.py` — Python implementations, FLOP accounting, timing
- `steps/` — the six checkpoints
- `tests/` — 36 tests (correctness on odd shapes; the speed claims as assertions)
- `bench/roofline.py` + `results.json` — the measured table
- `NOTES.md` — why each change works, and where the remaining gap lives

## Videos

Episode scripts live in [`video/topics/t16a/`](../../../video/topics/t16a/).
