---
topic: t16a-e1
episode: 1
title: Why matmul is everything
voice: am_michael
speed: 0.85
runtime_target_minutes: 10
paper: roofline model (Williams, Waterman & Patterson 2009)
---

## s1 · TitleCard

```props
{"title": "Matrix multiplication, part one", "subtitle": "Same flops, same compiler flags, 36x apart. The difference is the order you touch memory.", "topicId": "T16A", "episode": "Episode 1"}
```

Phase one, topic sixteen A. Matrix multiplication on a CPU. There is a GPU half of this topic in phase seven, and it will make no sense without this one. Here is the claim we are going to establish, and I want it up front so you can disbelieve it for the next ten minutes. The same two billion floating point operations, written five ways, compiled with identical flags, run thirty six times apart. Not because of the language. Not because of the compiler. Because of the order they touch memory.

## s2 · ConceptScene

```props
{"eyebrow": "the premise", "title": "Everything is a matmul", "body": "A transformer forward pass is matmuls with a little glue. Attention is two of them. The feed-forward block is two more. Fine-tuning, inference, embedding, retrieval — all of it bottoms out here.", "points": ["Well over 90% of the flops in a transformer are in dense matrix multiplies.", "Which is why hardware is designed around this one kernel: tensor cores exist for it.", "Which is why 'how fast is your model' is mostly 'how well is your matmul scheduled'.", "And why this is the literacy every kernel in Phases 6 and 7 assumes."], "aside": "You will not write the matmul your models use. You will spend years reasoning about its behaviour."}
```

Start with why this topic exists at all. A transformer forward pass is matrix multiplies with a small amount of glue. Attention is two of them. The feed forward block is two more. Fine tuning, inference, embedding, retrieval, all of it bottoms out in the same kernel. Well over ninety percent of the arithmetic in a transformer is dense matrix multiplication. That is why the hardware is designed around this one operation. Tensor cores exist for it specifically. It is why the question how fast is your model is mostly the question how well is your matmul scheduled. And it is why this is the literacy that every kernel in phases six and seven quietly assumes. You are almost certainly never going to write the matmul your models actually use. You are going to spend years reasoning about its behaviour, and that is a different skill which starts here.

## s3 · ChartScene

```props
{"eyebrow": "the starting point", "title": "Pure Python, and what it implies", "kind": "bar", "bars": [{"label": "n=128 measured", "value": 0.087, "color": "#7a8aa0", "note": "87 ms"}, {"label": "n=1024 implied", "value": 44.8, "color": "#e0b04e", "note": "45 seconds"}, {"label": "n=4096 implied", "value": 2867, "color": "#e05e6b", "note": "48 minutes"}], "logScale": true, "yLabel": "seconds", "caption": "0.05 GFLOP/s from the interpreter. Cost grows as n cubed, so the implication is worse than it looks."}
```

We start where everybody starts. The triple loop, in pure Python. At a hundred and twenty eight by a hundred and twenty eight it takes eighty seven milliseconds, which is fifty megaflops per second. Nobody is surprised. But hold onto the implication, because the cost grows as n cubed. At a thousand by a thousand that same rate is forty five seconds. At four thousand it is forty eight minutes. A single layer. One forward pass. That is the scale of the problem, and it is why the rest of this episode exists.

## s4 · MathReveal

```props
{"eyebrow": "the accounting", "title": "Arithmetic intensity: the permission to be fast", "english": "Count the flops, count the bytes that must move, and divide. That ratio decides whether a kernel can be limited by arithmetic at all, or is doomed to wait on memory.", "equation": "AI = 2·m·n·k / ( (mk + kn + mn) · 8 bytes )", "code": "n =   64  ->    5.3 FLOPs per byte\nn =  256  ->   21.3\nn = 1024  ->   85.3\nn = 4096  ->  341.3", "note": "It grows with n. That growth is permission, not a guarantee — you only collect it if you reuse what you loaded.", "stageFrames": [10, 140, 280]}
```

Before optimising anything, do the accounting, because it tells you what is even possible. In English. Count the floating point operations, count the bytes that have to move if every matrix is read once, and divide. That ratio is called arithmetic intensity, and it decides whether a kernel can be limited by arithmetic at all or is doomed to sit waiting on memory. In symbols. Two m n k flops, over m k plus k n plus m n elements, times eight bytes each. And in numbers. At sixty four, five point three flops per byte. At two hundred and fifty six, twenty one. At a thousand, eighty five. At four thousand, three hundred and forty one. It grows with n, and that growth is the entire reason matmul can be made fast. But read the note carefully. It is permission, not a guarantee. You only collect that reuse if you actually use the bytes you loaded before they are evicted. The naive loop does not, which is the next scene.

## s5 · DiagramScene

```props
{"eyebrow": "the crime", "title": "The inner loop walks a column", "nodes": [{"id": "a", "label": "A row i", "sub": "contiguous, cheap", "x": 0.2, "y": 0.25, "w": 0.3, "color": "#5ac48c", "appearAt": 8}, {"id": "b", "label": "B column j", "sub": "stride = n x 8 bytes", "x": 0.2, "y": 0.72, "w": 0.3, "color": "#e05e6b", "appearAt": 26}, {"id": "line", "label": "64-byte cache line", "sub": "8 doubles fetched", "x": 0.62, "y": 0.72, "w": 0.26, "appearAt": 52}, {"id": "used", "label": "1 used, 7 discarded", "sub": "87% waste", "x": 0.9, "y": 0.72, "w": 0.2, "color": "#e05e6b", "appearAt": 72}], "edges": [{"from": "b", "to": "line", "appearAt": 60}, {"from": "line", "to": "used", "appearAt": 78}], "caption": "Every load of B in the i-j-k order pulls a fresh line and uses one eighth of it."}
```

Here is the crime, and it is worth seeing rather than being told. In the textbook i j k order, the innermost loop varies k. So it walks along a row of A, which is contiguous and cheap. And it walks down a column of B, where consecutive elements are a whole row apart. Now, memory does not move in doubles. It moves in cache lines, and a cache line is sixty four bytes, which is eight doubles. So every single load of B fetches eight doubles, uses one of them, and throws seven away. That is eighty seven percent of your memory bandwidth spent on data you never look at. And the compiler cannot fix it, because the compiler is not allowed to change the meaning of your loops.

## s6 · CodeWalkthrough

```props
{"eyebrow": "the fix", "title": "One swapped loop", "filename": "phases/p1/t16a-matmul-cpu/src/t16a_matmul/kernels.c", "code": "/* i-j-k : inner loop varies k, B strides by a full row */\nfor (int j = 0; j < n; j++) {\n    double acc = 0.0;\n    for (int k = 0; k < k_dim; k++)\n        acc += A[i*k_dim + k] * B[k*n + j];\n    C[i*n + j] = acc;\n}\n\n/* i-k-j : inner loop varies j, B and C both walk rows */\nfor (int k = 0; k < k_dim; k++) {\n    const double a = A[i*k_dim + k];\n    for (int j = 0; j < n; j++)\n        C[i*n + j] += a * B[k*n + j];\n}", "highlights": [{"at": 20, "lines": [2, 3, 4, 5, 6], "caption": "The accumulator lives in a register — which is nice, and not worth what it costs."}, {"at": 160, "lines": [5], "caption": "B[k*n + j] with k varying: a fresh cache line every iteration."}, {"at": 300, "lines": [10, 11, 12, 13], "caption": "Swap the loops. Now A[i][k] is a scalar hoisted out, and j varies innermost."}, {"at": 420, "lines": [13], "caption": "B and C both walk contiguously — prefetchable, and the compiler can vectorise it."}, {"at": 520, "lines": [13], "caption": "The price: the accumulator moves out of a register and into memory, once per k."}]}
```

And here is the entire fix. Two versions of the same arithmetic. On top, i j k. The accumulator lives in a register, which feels efficient, and B strides by a whole row on every iteration of the inner loop. Underneath, i k j. We hoist the element of A out as a scalar, because it does not change in the inner loop. And now j varies innermost, so both B and C walk contiguously along rows. The hardware prefetcher can see the pattern. The compiler can vectorise the loop into wide registers. There is a real price. The accumulator has moved out of a register and into C, so we now write to memory once per k rather than once per output. And it is still, overwhelmingly, worth it.

## s7 · ChartScene

```props
{"eyebrow": "measured", "title": "1024x1024, identical compiler flags", "kind": "bar", "bars": [{"label": "i-j-k naive", "value": 0.53, "color": "#e05e6b", "note": "4098 ms"}, {"label": "i-k-j", "value": 8.3, "color": "#e0b04e", "note": "one swapped loop"}, {"label": "blocked", "value": 12.2, "color": "#5ac48c", "note": "episode 2"}, {"label": "blocked + threads + reg tile", "value": 18.7, "color": "#5ac48c", "note": "episode 2"}, {"label": "OpenBLAS", "value": 135.0, "color": "#7a8aa0", "note": "~100% of peak"}], "yLabel": "GFLOP/s", "logScale": true, "caption": "All five hand-written kernels: -O3 -march=native -fopenmp. Handicapping the naive one would be a better number and a worse lesson."}
```

Here is the measurement. All five kernels, compiled with identical flags. Dash O three, march native, f open m p. I want to be explicit about that, because it would be very easy to compile the naive version at dash O zero and report a much prettier ratio. It would also teach you nothing, because then you would be measuring the optimiser rather than the algorithm. Naive i j k gets zero point five three gigaflops. The swapped loop gets eight point three. That is fifteen point eight times, from moving one line. Blocking, which is episode two, gets twelve point two. Blocking with threads and a register tile gets eighteen point seven. And OpenBLAS, at the far right, gets a hundred and thirty five.

## s8 · ConceptScene

```props
{"eyebrow": "the ceiling", "title": "Always know your 100%", "body": "A GFLOP/s number means nothing on its own. Compute the machine's ceiling by hand: cores times clock times FLOPs per cycle. Then every measurement becomes a percentage of something real.", "points": ["2 cores x 2.10 GHz x 32 FLOPs/cycle = 134 GFLOP/s.", "The 32 is AVX-512: 8 doubles per vector, 2 flops per FMA, 2 FMA units.", "OpenBLAS measures 135. It is at essentially 100% of peak.", "So our 18.7 is 14% of peak — and 'why 14%' is a far better question than 'how do I beat BLAS'."], "aside": "The useful interview question is never 'how many GFLOP/s'. It is 'what fraction of peak, and what is your peak?'"}
```

And now the most valuable habit in this entire topic. Always know what a hundred percent is. A gigaflops number on its own means nothing. So compute the ceiling by hand. Cores, times clock, times floating point operations per cycle. Two cores, at two point one gigahertz, times thirty two flops per cycle, is a hundred and thirty four gigaflops. Where does the thirty two come from. AVX five twelve holds eight doubles per vector, a fused multiply add is two flops, and the core has two of those units. Eight times two times two is thirty two. Now look back at the chart. OpenBLAS measured a hundred and thirty five. It is running at essentially one hundred percent of theoretical peak. Which means our best hand written kernel is at fourteen percent. And that reframes the goal completely. Beating OpenBLAS is not the goal and never was. Explaining the fourteen percent is the goal, and episode two is that explanation.

## s9 · RecapScene

```props
{"eyebrow": "episode one", "points": ["Over 90% of a transformer's flops are dense matmul — this is the kernel the hardware was designed for.", "Arithmetic intensity grows with n: that is permission to be compute-bound, not a guarantee.", "The naive i-j-k loop walks a column of B and discards 7 of every 8 doubles it fetches.", "Swapping two loops is worth 15.8x. Same flops, same flags, same result.", "Always compute your machine's peak by hand, so every number is a percentage of something real."], "ifSkipped": "Every kernel in Phases 6 and 7 — KV paging, quantization, Flash Attention — assumes you can reason about this.", "next": "Episode 2: cache blocking, threads, and where the remaining 7x to OpenBLAS actually lives."}
```

To recap. Over ninety percent of a transformer's arithmetic is dense matrix multiply, which is the kernel the hardware was designed around. Arithmetic intensity grows with n, and that is permission to be compute bound rather than a guarantee of it. The naive i j k loop walks a column of B and throws away seven of every eight doubles it fetches. Swapping two loops is worth fifteen point eight times, with the same flops, the same flags and the same result. And always compute your machine's peak by hand, so that every number you produce is a percentage of something real. Skip this and every kernel in phases six and seven, the KV cache pager, the quantiser, Flash Attention, is a recipe you follow rather than a design you understand. Next episode. Cache blocking, threads, and an honest account of where the remaining seven times to OpenBLAS actually lives.
