---
topic: t16a-e2
episode: 2
title: Cache lines make it fast
voice: am_michael
speed: 0.85
runtime_target_minutes: 11
paper: roofline model; BLIS micro-kernel design (Van Zee & van de Geijn 2015)
---

## s1 · TitleCard

```props
{"title": "Matrix multiplication, part two", "subtitle": "Blocking, threads, register tiles — and an honest account of the gap that is left.", "topicId": "T16A", "episode": "Episode 2"}
```

Episode two. Last time we fixed the stride and got fifteen times. Today we fix the volume, add threads, add a register tile, and then spend the last third of the episode being honest about the seven times that remains between us and OpenBLAS. That last part matters more than the optimisations, because knowing where a gap lives is a transferable skill and a tuned constant is not.

## s2 · ConceptScene

```props
{"eyebrow": "the second problem", "title": "The right stride, the wrong volume", "body": "The i-k-j loop touches memory in the right pattern. It still streams the entire B matrix past the cache once for every row of A.", "points": ["n = 1024, float64: B is 8 MB. It does not fit in any cache on this machine.", "One row of A consumes all of B. There are 1024 rows.", "So the kernel moves roughly 8 GB, to do 2 GFLOP of arithmetic.", "Arithmetic intensity of 85 available; we are collecting about 0.25."], "aside": "Stride is about how you touch a cache line. Volume is about how many times you touch the same data. They are different bugs with different fixes."}
```

So we fixed the stride. Here is the second problem, which is a completely different one. The i k j loop touches memory in the right pattern, and it still streams the entire B matrix past the cache once for every single row of A. Do the arithmetic. At n equals one thousand and twenty four, in double precision, B is eight megabytes. That fits in no cache on this machine. One row of A consumes all of B, and there are one thousand and twenty four rows. So the kernel moves something on the order of eight gigabytes of traffic to perform two gigaflops of arithmetic. Remember the arithmetic intensity from last episode. Eighty five flops per byte available. We are collecting about a quarter of one. Stride is about how you use a cache line once you have it. Volume is about how many times you go and get the same data. Different bugs, different fixes.

## s3 · DiagramScene

```props
{"eyebrow": "the fix", "title": "Blocking: keep a panel of B resident", "nodes": [{"id": "a", "label": "A tile", "sub": "MC x KC rows", "x": 0.16, "y": 0.5, "w": 0.2, "h": 0.5, "color": "#5ac48c", "appearAt": 8}, {"id": "b", "label": "B panel", "sub": "KC x NC ~ 256 KB", "x": 0.46, "y": 0.5, "w": 0.22, "h": 0.5, "color": "#e0b04e", "appearAt": 26}, {"id": "cache", "label": "L2", "sub": "B panel lives here", "x": 0.72, "y": 0.28, "w": 0.16, "appearAt": 52}, {"id": "c", "label": "C tile", "sub": "accumulated in place", "x": 0.9, "y": 0.68, "w": 0.18, "appearAt": 70}], "edges": [{"from": "b", "to": "cache", "label": "loaded once", "appearAt": 58}, {"from": "a", "to": "c", "label": "MC rows reuse it", "appearAt": 76}], "caption": "Load a panel of B into L2, then push many rows of A through it before evicting."}
```

The fix is blocking, and the picture is simple. Instead of streaming all of B for each row of A, take a panel of B. Sized so that it fits in the cache level you are aiming at. Load it once. Then push many rows of A through that resident panel before you let it go. The traffic drops by roughly the number of rows you reuse it for. In this implementation there are three block sizes. M C, how many rows of A per tile. K C and N C, the two dimensions of the B panel. The defaults aim the B panel at a quarter of a megabyte, which is roughly L2 on this machine. And the crucial word in that sentence is roughly, which the next scene is about.

## s4 · ChartScene

```props
{"eyebrow": "measured", "title": "The block-size sweep is flat, and that is the finding", "kind": "bar", "bars": [{"label": "64/64/128 (64 KB)", "value": 12.08, "color": "#7a8aa0"}, {"label": "128/128/256 (256 KB)", "value": 13.55, "color": "#5ac48c", "note": "best"}, {"label": "256/128/256 (256 KB)", "value": 13.13, "color": "#5ac48c"}, {"label": "128/256/512 (1 MB)", "value": 12.92, "color": "#7a8aa0"}, {"label": "64/256/1024 (2 MB)", "value": 10.53, "color": "#e0b04e", "note": "panel too big"}], "yLabel": "GFLOP/s", "caption": "Worst over best across six plausible tilings: 1.27x. Block size is not a magic constant."}
```

Now, everyone expects a block size sweep to reveal a magic number. Here is the actual sweep, six plausible tilings at n equals one thousand and twenty four. The best is one twenty eight, one twenty eight, two fifty six, giving thirteen point five five. The worst is sixty four, two fifty six, one thousand and twenty four, giving ten point five three, and that one is worst because its B panel is two megabytes and no longer fits. Worst over best across the whole table is one point two seven times. That is the finding, and it is more useful than a magic number would have been. Block size is not a constant you tune to three significant figures. It is the sentence, make the panel roughly L2 sized. Anything in that neighbourhood is within twenty percent, and the twenty percent is not where your remaining seven times is hiding.

## s5 · ChartScene

```props
{"eyebrow": "measured", "title": "Reading the memory hierarchy off your own machine", "kind": "line", "series": [{"label": "read+write bandwidth", "color": "#5ac48c", "values": [41.5, 54.2, 70.9, 81.1, 44.6, 43.2, 33.8]}], "xLabel": "working set: 16 KB -> 64 MB, x4 per step", "yLabel": "GB/s", "caption": "In-place scale over arrays of growing size. The cliff between 1 MB and 4 MB is a cache level ending."}
```

And here is how you find that neighbourhood without reading a spec sheet. Take an array, scale it in place, and time it. Two bytes move per array byte, one read and one write. Now grow the array. At sixteen kilobytes we get forty one gigabytes per second, and that row understates because the per call overhead is a real fraction of the time. At two hundred and fifty six kilobytes, seventy one. At one megabyte, eighty one, the peak. And then at four megabytes it falls off a cliff to forty four, and by sixty four megabytes it is down to thirty four, which is main memory. Read the cliffs, not the absolute numbers. The cliff between one and four megabytes is a cache level ending, and it is precisely why the sweep preferred a panel of a quarter of a megabyte. This chart took nine lines of NumPy and it is worth more than any spec sheet, because it describes the machine you are actually running on today.

## s6 · CodeWalkthrough

```props
{"eyebrow": "threads", "title": "Parallelise over j, and you need no locks at all", "filename": "src/t16a_matmul/kernels.c", "code": "#pragma omp parallel for schedule(static)\nfor (int jj = 0; jj < n; jj += nc) {\n    const int jmax = (jj + nc < n) ? jj + nc : n;\n    for (int kk = 0; kk < k_dim; kk += kc) {\n        for (int ii = 0; ii < m; ii += mc) {\n            for (int i = ii; i < imax; i++) {\n                double *restrict crow = &C[(size_t)i * n];\n                for (int k = kk; k < kmax; k++) {\n                    const double a = A[(size_t)i * k_dim + k];\n                    const double *restrict brow = &B[(size_t)k * n];\n                    for (int j = jj; j < jmax; j++)\n                        crow[j] += a * brow[j];\n}   }   }   }   }", "highlights": [{"at": 20, "lines": [1, 2], "caption": "Parallel over jj: each thread owns a disjoint column band of C."}, {"at": 150, "lines": [2, 3], "caption": "Disjoint writes mean no locks, no atomics, and no reduction."}, {"at": 280, "lines": [4], "caption": "Parallelising over k instead would need a reduction — every thread writing the same C."}, {"at": 400, "lines": [7, 10], "caption": "'restrict' promises no aliasing, which is what lets the compiler hoist 'a' and vectorise the store."}]}
```

Threads next, and the interesting part is choosing which loop to parallelise. We parallelise over j j, the column tiles. Each thread owns a disjoint band of columns of C. Disjoint writes mean no locks, no atomics, and no reduction step at the end. Compare that with parallelising over k, which would have every thread accumulating into the same elements of C, and would need either atomics or per thread buffers and a merge. Same arithmetic, dramatically different engineering, decided entirely by which index you hand to OpenMP. And notice the restrict keyword on the two pointers. That is a promise to the compiler that these arrays do not overlap. Without it the compiler must assume that writing to C might change B, which forbids hoisting the value of A out of the inner loop and forbids vectorising the store. One keyword, and a large fraction of the optimisation.

## s7 · Callout

```props
{"kind": "insight", "heading": "Two cores gave 1.4x, not 2x — and that is the expected answer", "body": "The kernel is partly memory-bound, and the two cores share one memory system. When a thread count does give you linear scaling on a memory-heavy kernel, that is the surprising result, not this.", "code": "blocked, 1 thread   :  181 ms   12.2 GFLOP/s\nblocked, 2 threads  :  125 ms   17.2 GFLOP/s   (1.45x)\n+ 4-row register tile: 115 ms   18.7 GFLOP/s"}
```

Threading gets us from a hundred and eighty one milliseconds to a hundred and twenty five. One point four five times on two cores, not two times, and I want to name that rather than quietly move past it. The kernel is partly memory bound, and the two cores share a single memory system, so adding a core adds arithmetic capacity and adds no bandwidth. Sub linear scaling on a memory heavy kernel is the expected outcome. If you ever see linear scaling on a kernel like this, that is the surprising result and it usually means you were more compute bound than you thought. Adding a four row register tile takes us to a hundred and fifteen milliseconds and eighteen point seven gigaflops, which is where our hand written ladder ends.

## s8 · ConceptScene

```props
{"eyebrow": "the honest part", "title": "Where the remaining 7x lives", "body": "We are at 18.7 GFLOP/s against OpenBLAS's 135, which is essentially 100% of this machine's peak. The gap is not mysterious, and none of it is 'C is slow'.", "points": ["C is not held in registers: every FMA carries a load and a store of C.", "No packing: BLIS-style kernels copy A and B panels into contiguous, pre-swizzled buffers.", "Hand-written assembly: per-microarchitecture, software-pipelined, with explicit prefetch.", "Our register tile is 4x1. A real micro-kernel is more like 8x6, held entirely in vector registers across the whole k loop."], "aside": "The single biggest remaining win is the first one — and it is exactly what the GPU pass in Phase 7 calls 'accumulate in registers'."}
```

And now the part I think is worth the most. We are at eighteen point seven, OpenBLAS is at one hundred and thirty five, and the honest question is where the difference goes. None of it is C being slow. First, and biggest. Our kernel does not hold C in registers. Every fused multiply add carries a load of C and a store back to C. A real micro kernel holds a small tile of C, say eight by six doubles, entirely in vector registers, accumulates across the whole k loop, and stores it exactly once at the end. Second, packing. BLIS style kernels copy the A and B panels into contiguous pre swizzled buffers, so the micro kernel sees perfectly aligned unit stride data with no page table pressure. Third, hand written assembly, per micro architecture, software pipelined, with explicit prefetch instructions. And the reason the first one matters most to you is that it is precisely what the GPU pass in phase seven calls accumulating in registers. Same sentence. Different memory hierarchy.

## s9 · RecapScene

```props
{"eyebrow": "topic recap", "points": ["Stride and volume are different bugs: loop order fixes one, blocking fixes the other.", "Block size is 'roughly L2-sized' — the sweep is flat within 1.27x, so stop tuning it.", "Measure your own memory hierarchy in nine lines; read the cliffs, not the numbers.", "Parallelise over the index whose outputs are disjoint, and you need no synchronisation at all.", "22.6x from blocking alone, 35.6x with threads and a register tile — and 14% of peak, for reasons you can name."], "ifSkipped": "T16B, T45B and T7 all rebuild this reasoning on a GPU. Shared memory plays the role of L2 and the sentences are unchanged.", "next": "Next topic: T45A — softmax, and the one-pass trick Flash Attention is built on."}
```

To recap the topic. Stride and volume are different bugs. Loop order fixes one, blocking fixes the other, and confusing them wastes a lot of time. Block size is roughly L2 sized, the sweep is flat within one point two seven times, so stop tuning it and go find a real bottleneck. Measure your own memory hierarchy in nine lines of NumPy and read the cliffs rather than the absolute numbers. Parallelise over whichever index makes the outputs disjoint, and you will need no synchronisation whatsoever. And the headline. Twenty two point six times from blocking alone, thirty five point six with threads and a register tile, landing at fourteen percent of this machine's theoretical peak for reasons you can now list on demand. Phase seven rebuilds this entire chain of reasoning on a GPU, where shared memory plays the part of L2 and the sentences do not change. Next topic is forty five A. Softmax. Which sounds like a small thing and turns out to contain the single trick that Flash Attention is built on.
