---
topic: p0.2
episode: 1
title: NumPy thinking
voice: am_michael
speed: 0.85
runtime_target_minutes: 11
paper: none — the reference implementation is pandas rolling/ewm
---

## s1 · TitleCard

```props
{"title": "NumPy as your new LINQ", "subtitle": "Describing a computation is not the same as deleting its loop.", "topicId": "P0.2", "episode": "Episode 1"}
```

Phase zero, topic two. NumPy as your new LINQ. LINQ taught you to describe what you want instead of writing the loop that gets it. NumPy asks for the same description and then actually removes the loop. That is a stronger promise, and it comes with a sharper set of edges. We are going to build a technical indicator suite. Simple and exponential moving averages, Wilder's smoothing, the relative strength index, Bollinger bands, volume weighted average price, and average true range. No Python level loop will touch a single bar of data. But the indicators are the vehicle. The subject is array thinking.

## s2 · ArchitectureMap

```props
{"eyebrow": "where we are", "highlight": ["data"], "caption": "P0.2 builds the feature table the research surface and the feature store both read."}
```

Here is where this lands in AlphaDesk. The data block. By the end of the episode we have a feature table over the price history, computed per symbol, with no lookahead. In phase three, topic forty eight makes that table point in time correct, and the two tower recommender in topic forty two reads from it. So this is the first piece of the desk's data layer, and the discipline we establish here about causality is the discipline the whole desk inherits.

## s3 · ConceptScene

```props
{"eyebrow": "why loops die", "title": "A Python list is not a double[]", "body": "A List<double> in .NET is a contiguous block of unboxed doubles. A Python list is an array of pointers to individually heap-allocated, boxed float objects.", "points": ["Eight bytes of payload wrapped in about thirty-two bytes of object header.", "Every addition is a type lookup, a virtual dispatch and an allocation.", "np.ndarray is what you assumed the list already was: contiguous, unboxed, with the loop living in C.", "So the 70x is not clever optimisation. It is the interpreter getting out of the way."], "aside": "1,000,000 float64: python list ~32 MB · ndarray 8 MB"}
```

Start with why the loop is slow, because the answer is not the one people assume. A list of doubles in dot NET is a contiguous block of unboxed doubles with a count on the front. A Python list is not that. It is an array of pointers to individually heap allocated, boxed float objects. Eight bytes of actual payload wrapped in roughly thirty two bytes of object header, scattered across memory. Every single addition is a dynamic type lookup, a virtual dispatch, and an allocation for the result. A NumPy array is what you assumed the list already was. One contiguous block of unboxed float sixty four, with the loop living down in C. A million float sixty fours occupy about thirty two megabytes as a Python list and exactly eight as an array. So the seventy times speed up you are about to see is not clever optimisation. It is the interpreter getting out of the way. Two consequences follow. Micro optimising the body of a Python loop is nearly pointless, because you have to remove the interpreter from the loop entirely. And a NumPy operation on ten elements is slower than the plain Python equivalent, because the per call overhead dominates. Vectorisation is a scale play.

## s4 · ConceptScene

```props
{"eyebrow": "broadcasting", "title": "The loop you do not write", "body": "Compare shapes right to left. Two dimensions are compatible when they are equal, or when one of them is one. A missing leading dimension counts as one.", "points": ["prices (5, 260)  ×  weights[:, None] (5, 1)  →  (5, 260)", "keepdims=True is what keeps a reduction's result broadcastable.", "Half of all 'could not be broadcast' errors are a missing keepdims.", "Broadcasting does not allocate the inputs. It does allocate the output — a (10000,1) plus a (1,10000) is 800 megabytes."], "aside": "There is no C# analogue. The nested loop is the habit to unlearn."}
```

Broadcasting is the rule that lets arrays of different shapes combine without copying, and it is the loop you do not write. Compare shapes from the right. Two dimensions are compatible when they are equal, or when one of them is one. A missing leading dimension counts as one. So a five by two hundred and sixty matrix of prices, times a five by one column of weights, gives you back five by two hundred and sixty, with a weight applied per row and no tiling anywhere. The keyword that makes this work in practice is keep dims. A reduction over an axis normally drops that axis, which then fails to line up with the array you wanted to divide. Keep dims equals true leaves a length one axis in place, and it broadcasts. Roughly half of all could not be broadcast errors are a missing keep dims. And here is the trap. Broadcasting does not allocate the inputs. It absolutely does allocate the output. A ten thousand by one array plus a one by ten thousand array is two arrays of eighty kilobytes each producing a result of eight hundred megabytes. Read the output shape before you run the line.

## s5 · MathReveal

```props
{"eyebrow": "the indicator", "title": "The simple moving average", "english": "The average of the last w closing prices, recomputed at every bar.", "equation": "SMA_t = (1/w) · Σ_{k=t-w+1..t} close_k", "code": "sliding_window_view(a, w).mean(axis=-1)", "note": "sliding_window_view returns a view: (n-w+1, w) with zero bytes copied.", "stageFrames": [10, 120, 240]}
```

Take the simplest indicator there is and write it three ways. In words. The average of the last w closing prices, recomputed at every bar. In symbols. S M A at time t equals one over w, times the sum of the closes from t minus w plus one, up to t. And in code, one line. Sliding window view of the array, then take the mean along the last axis. That function is the important part. It gives you a two dimensional view of shape n minus w plus one, by w. Every row is one window. And it is a view, not a copy. Zero bytes are moved. It is built out of stride tricks, which means it reinterprets the existing memory with a different shape and step size. The reduction then runs entirely in C.

## s6 · ChartScene

```props
{"eyebrow": "three implementations", "title": "The same average, three ways", "kind": "bar", "bars": [{"label": "python loop", "value": 457, "color": "#e05e6b", "note": "ms, scaled"}, {"label": "cumsum", "value": 3.7, "color": "#e0b04e", "note": "ms — but see below"}, {"label": "sliding_window", "value": 6.4, "color": "#5ac48c", "note": "ms — shipped"}, {"label": "pandas rolling", "value": 4.7, "color": "#4ec9d6", "note": "ms — the reference"}], "logScale": true, "caption": "20-day SMA over 200,000 bars. Log scale. The fastest one is not the one that ships."}
```

Now the timing, on two hundred thousand bars with a twenty day window, and note this is a log scale. The nested Python loop takes four hundred and fifty seven milliseconds. The sliding window version takes six point four, which is about seventy two times faster. Pandas rolling, which is our reference implementation, takes four point seven. And then there is the middle bar. The cumulative sum trick takes three point seven milliseconds, making it the fastest of all of them. It is also the one we did not ship. That deserves an explanation.

## s7 · Callout

```props
{"kind": "warning", "heading": "The cumsum trick loses the bits you wanted", "body": "A rolling sum can be computed in linear time by differencing a running total. But that subtracts two large, nearly-equal numbers. On two million bars at an index level of one billion, the running total reaches 2e15 — and float64 carries about sixteen significant digits, so the difference throws away exactly the low-order bits you were trying to keep. Measured error: 1.3e-2. On rupees. On a price.", "code": "csum = np.concatenate([[0.0], np.cumsum(x)])\nout[w-1:] = (csum[w:] - csum[:-w]) / w   # O(n), and lossy"}
```

The trick is to build one running total and then subtract the total at the start of the window from the total at the end. That is linear time instead of linear times window. It is also catastrophic cancellation. You are subtracting two large, nearly equal numbers. On two million bars at an index level of one billion, the running total reaches two times ten to the fifteen. Float sixty four carries about sixteen significant digits. So the difference throws away exactly the low order bits you were trying to keep. The measured error is one point three times ten to the minus two. That is rupees, on a price. Now, on one year of two digit prices this error is completely invisible, and that is precisely what makes it dangerous. The fastest formulation and the numerically sound one are different, and the whole skill is knowing which one you picked.

## s8 · ConceptScene

```props
{"eyebrow": "the harder case", "title": "Some things are scans, not reductions", "body": "A simple moving average is a reduction over independent windows, so it vectorises trivially. An exponential moving average is a scan: every output depends on the previous output.", "points": ["y[t] = a · x[t] + (1 - a) · y[t-1]", "There is a closed form, and it is the 'vectorised EMA' every blog post shows.", "It divides by (1-a) to the power k, which passes float64's limit around bar 4,200 for a 12-day span.", "Exact to 1e-14 right up to the cliff. Then the entire tail is NaN, all at once."], "aside": "scipy.signal.lfilter — the same loop, moved into C. Thirty times faster and stable."}
```

Now a harder case, and the one that teaches the real lesson. A simple moving average is a reduction over independent windows, so it vectorises trivially. An exponential moving average is not. It is a scan. Y at t equals alpha times x at t, plus one minus alpha times y at t minus one. Every output depends on the previous output. There is a closed form. You can expand the recursion into a cumulative sum of x divided by one minus alpha to the power k. That is the vectorised E M A that every blog post shows you, and it is a landmine. That divisor grows exponentially. For a twelve day span it passes float sixty four's maximum of one point eight times ten to the three hundred and eight at around bar four thousand two hundred. Here is the cruel part. Up to that point the answer is exact to fourteen decimal places. One bar later, the entire output is not a number. A test on one year of data will never see it. Seventeen years of daily data will. The professional answer is to stop trying to turn a scan into a reduction. Scipy signal l filter implements exactly this one pole recursive filter, in C. Same algorithm, same sequential dependency, thirty times faster than the Python loop, and numerically stable. That is the shape of the skill. Recognise whether you have a reduction, a scan, or a convolution, and reach for the primitive that already implements it.

## s9 · Callout

```props
{"kind": "gotcha", "heading": "Wilder's smoothing is not an EMA", "body": "Wilder's fourteen-period average uses alpha = 1/14. A fourteen-span EMA uses alpha = 2/15 — nearly double the responsiveness. Almost every 'my RSI doesn't match TradingView' report is this, or the seeding convention, or ddof on the Bollinger standard deviation. None of them is wrong. They are conventions, and they must be pinned explicitly in code.", "code": "wilder(14):  alpha = 1/14      = 0.0714\nema(14):     alpha = 2/(14+1) = 0.1333\nema(27)  ≈  wilder(14)"}
```

One specific gotcha, because it will cost you an afternoon otherwise. Wilder's smoothing is not an exponential moving average. Wilder's fourteen period average uses an alpha of one over fourteen, which is zero point zero seven one. A fourteen span E M A uses two over fifteen, which is zero point one three three. Nearly double the responsiveness. An E M A with a span of twenty seven is the closer match. Essentially every my R S I does not match the other library report comes down to one of three things. That alpha. Or the seeding convention, meaning whether you start the recursion from the simple mean of the first n values or from the very first value. Or the degrees of freedom on the Bollinger standard deviation, where the original definition uses the population version and the pandas default uses the sample version. None of these is wrong. They are conventions. The lesson is to pin the convention explicitly in code, and to write the reference implementation before the fast one.

## s10 · MathReveal

```props
{"eyebrow": "einsum", "title": "The subscript string, and why it matters", "english": "Name every axis. A repeated name is contracted — summed over. A name missing from the output is summed away.", "equation": "\"bhqd,bhkd->bhqk\"", "code": "scores = np.einsum(\"bhqd,bhkd->bhqk\", q, k) / np.sqrt(depth)", "note": "batch · head · query · key · depth. That line is scaled dot-product attention.", "stageFrames": [10, 130, 250]}
```

Last idea, and it is the one that pays off two phases from now. Einsum. Name every axis with a letter. A repeated letter is contracted, meaning summed over. A letter missing from the output is summed away. That is the entire rule. Matrix multiply is i j comma j k, arrow, i k. Row sums are i j arrow i. The diagonal is i i arrow i. A quadratic form, weights transpose sigma weights, which is portfolio variance, is i comma i j comma j arrow nothing. And then there is the string on screen. B h q d, comma, b h k d, arrow, b h q k. Batch, head, query, key, depth. Keep batch and head, contract over depth, and pair every query position with every key position. That single line is the numerator of scaled dot product attention. It is topic four, in phase two. Being able to read it fluently is the difference between following that topic and transcribing it.

## s11 · CodeWalkthrough

```props
{"eyebrow": "the finance gotcha", "title": "The test almost nobody writes", "filename": "phases/p0/p0-2-numpy-as-linq/tests/test_indicator_properties.py", "code": "def test_no_lookahead(walk):\n    \"\"\"Today's value cannot depend on tomorrow.\"\"\"\n    cut = 300\n    for name, fn in [(\"sma\", lambda x: sma(x, 20)),\n                     (\"ema\", lambda x: ema(x, 12)),\n                     (\"rsi\", lambda x: rsi(x, 14)),\n                     (\"macd\", lambda x: macd(x)[0])]:\n        full = fn(walk)[:cut]\n        partial = fn(walk[:cut])\n        assert np.array_equal(full, partial, equal_nan=True), f\"{name} peeks\"", "highlights": [{"at": 30, "lines": [1, 2, 3], "caption": "Five lines. It catches the bug that quietly inflates every backtest you will ever write."}, {"at": 200, "lines": [8, 9], "caption": "Truncate the input, recompute, and demand bit-identical earlier values."}, {"at": 330, "lines": [10], "caption": "Centred windows, backfill, and a groupby over the whole history all fail this silently."}]}
```

I want to end on a test rather than on a feature, because this is the one that matters most and the one almost nobody writes. An indicator at time t must depend only on data up to time t. To check that, you truncate the series, recompute, and demand that every earlier value comes back bit identical. Five lines. It catches centred rolling windows, it catches backward fill, and it catches a group by transform computed over the whole history. All three of those fail silently, produce beautiful backtests, and are completely worthless. There is a second one in the same file worth mentioning. A twenty day average computed over a concatenated multi symbol frame will blend the last nineteen bars of one issuer into the first bars of the next. There is a test that proves the group by is load bearing rather than decorative. Group first. Always.

## s12 · RecapScene

```props
{"eyebrow": "P0.2", "title": "Recap", "points": ["The speed-up is the interpreter leaving the loop, not NumPy being clever — so remove the interpreter entirely or do not bother.", "Broadcasting is the loop you do not write; keepdims makes reductions line up, and the output is what gets allocated.", "Reduction, scan or convolution — knowing which one you have is what tells you whether to vectorise or to reach for lfilter.", "The fastest formulation and the numerically sound one are frequently different. Measure, at scale, against an independent reference."], "ifSkipped": "Phase 1 assumes you read shapes fluently. Without broadcasting and einsum, an autograd bug in T31 and a transformer in T4 look like framework problems when they are array problems.", "next": "P0.3 · PyTorch tensors and the training loop"}
```

Four things to take away. The speed up is the interpreter leaving the loop, not NumPy being clever, so remove the interpreter entirely or do not bother at all. Broadcasting is the loop you do not write, keep dims is what makes reductions line up again, and the output is the thing that gets allocated. Reduction, scan, or convolution. Knowing which one you have is what tells you whether to vectorise or to reach for a filter primitive. And the fastest formulation and the numerically sound one are frequently different, so measure, at scale, against an independent implementation. What breaks if we skip this. Topic zero point three and all of phase one assume you read shapes fluently. An autograd bug in topic thirty one shows up as a shape mismatch in a backward pass. A transformer in topic four is ninety per cent shape bookkeeping. Without broadcasting and einsum, every later topic will look like a framework problem when it is really an array problem. Next episode. PyTorch tensors, and the training loop.
