# T16A · CPU matmul — Notes

## The one idea

Every kernel here does **exactly the same arithmetic**: 2·n³ flops, same
operands, same result to within float non-associativity. The 36× spread between
the slowest and fastest comes entirely from the order they touch memory. That
is the lesson, and it generalises to every kernel in Phases 6 and 7.

## Why matmul specifically

Arithmetic intensity — FLOPs per byte at perfect reuse — grows with n:

| n | 64 | 256 | 1024 | 4096 |
|:--|--:|--:|--:|--:|
| FLOPs/byte | 5.3 | 21.3 | 85.3 | 341.3 |

That growth is the *permission* to be compute-bound. It is not a guarantee. You
only collect the reuse if you actually use the bytes you loaded before evicting
them, and the naive loop does not.

## Walking through the ladder

**i-j-k → i-k-j: 15.8×, from one swapped line.** In i-j-k the inner loop is
`acc += A[i][k] * B[k][j]` with k varying — B strides by a whole row per step,
so every load is a fresh 64-byte line of which 8 bytes are used. Swap to i-k-j
and the inner loop varies j: `C[i][j] += a * B[k][j]`, both contiguous, both
vectorisable. The price is that the accumulator moves from a register into C.

**Blocking: another 1.4×.** i-k-j has the right *stride* but the wrong *volume*:
for each row of A it streams all of B past the cache. At n=1024 that is 8 MB per
row, ~8 GB total. Tiling reuses a resident panel of B across many rows of A.

The block sweep is deliberately part of the deliverable, because the honest
finding is that the table is flat: worst/best across six plausible tilings is
**1.27×**. Block size is not a magic constant, it is "roughly L2-sized", and the
bandwidth curve tells you where that is.

**Threads: 1.4×on two cores.** Column tiles of C are disjoint, so
`#pragma omp parallel for` over `jj` needs no reduction and no locks. Sub-linear
scaling on two cores is expected — the kernel is partly memory-bound, and the
two cores share the same memory system.

**Register tiling: 1.1×.** Four rows of A per pass, so one load of `B[k][j]`
feeds four FMAs. Small here, and the reason it is small is instructive: our
inner loop is still bound by C traffic, not B traffic.

## Where the remaining 7× to OpenBLAS lives

We reach 18.7 GFLOP/s; OpenBLAS reaches 135. The gap is not mysterious:

1. **C is not held in registers.** Our kernel writes `C[i][j] +=` on every k
   iteration, so every FMA carries a load and a store. A real micro-kernel holds
   a small tile of C (say 8×6 doubles) in vector registers across the *entire* k
   loop and stores it once at the end. This is the single biggest remaining win.
2. **No packing.** BLIS-style kernels copy A and B panels into contiguous,
   pre-swizzled buffers so the micro-kernel sees perfectly aligned unit-stride
   data with no TLB pressure.
3. **Hand-written assembly.** OpenBLAS's inner kernels are per-microarchitecture
   assembly with software pipelining and explicit prefetch, not compiler output.

Estimating the ceiling is worth doing once by hand: 2 cores × 2.10 GHz × 32
FLOPs/cycle (AVX-512: 8 doubles × 2 flops × 2 FMA units) ≈ **134 GFLOP/s**.
OpenBLAS measures 135. It is at essentially 100% of peak, which is why "beat
BLAS" is not the goal and "understand the 14%" is.

## Gotchas

- **Fair flags.** Everything is compiled `-O3 -march=native -fopenmp`. Compiling
  the naive kernel at `-O0` would report a prettier ratio and teach nothing.
- **`restrict` matters.** Without it the compiler must assume A, B and C may
  alias, and cannot keep `a` in a register across the inner loop or vectorise
  the store. It is the C equivalent of promising there is no aliasing.
- **Tails.** Every blocked loop has a remainder when the dimension is not a
  multiple of the tile. The tests use sizes like 37×53×41 and 65×33×129 on
  purpose; that is where blocked kernels break.
- **Bit-exactness with BLAS is not a goal.** A different summation order over
  1024 terms moves the last two digits. Float addition is not associative.
  Compare with `rtol=1e-8`, not `==`.
- **Timing: best-of-N, not mean.** On a shared machine the slow runs measure the
  neighbours. Best-of-N measures the kernel.
- **First-touch pages.** The first call to a kernel also pays for page faults on
  the freshly allocated C. `time_call` runs one warm-up call before timing.

## Carry-forward

- Phase 7 (T16B) does this again on a GPU. "Tile so the data stays in the fast
  memory, accumulate in registers, then store once" is *the same sentence*, with
  shared memory playing the role of L2 and registers playing the role of
  registers.
- T7 (Flash Attention) is this thinking applied to softmax(QKᵀ)V: the win is not
  fewer flops, it is never materialising the n×n intermediate.
- When someone quotes a GFLOP/s number in an interview, the useful follow-up is
  "what fraction of peak, and what is your peak?" This topic is where you learn
  to answer that about your own code.
