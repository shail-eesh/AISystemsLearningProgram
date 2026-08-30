---
topic: p0.1
episode: 1
title: Python for .NET architects
voice: am_michael
speed: 0.85
runtime_target_minutes: 10
paper: none — this episode is a translation layer, not a paper
---

## s1 · TitleCard

```props
{"title": "Python for the .NET veteran", "subtitle": "The four guarantees that move from the compiler into your discipline.", "topicId": "P0.1", "episode": "Episode 1"}
```

Welcome to the AI Systems Forge. This is Phase zero, topic one. Python for the dot NET veteran. You already know how to design systems. You know domain models, you know dependency injection, you know why money is a decimal and never a double. None of that is being re-taught here. What this episode does is different. It maps the constructs you already reach for onto their Python counterparts, and then it spends most of its time on the gaps between them. Because the gaps are where the bugs live.

## s2 · ArchitectureMap

```props
{"eyebrow": "where we are", "highlight": ["orders"], "caption": "P0.1 builds the paper order book every later topic places orders into."}
```

Before any code, here is where we are in the build. Every topic in this course plugs into one system. It is called AlphaDesk, and it is a fictional educational simulation. No real orders, no real money, no brokerage connectivity. Today we build the block that is lit up. The order workflow. Order, fill, position, portfolio. In phase four, a reasoning agent will construct an order ticket in natural language and place it into exactly this book. So the domain model we write in the next ten minutes is not a throwaway exercise. It is load bearing for the rest of the course.

## s3 · ConceptScene

```props
{"eyebrow": "the premise", "title": "Same vocabulary, different guarantees", "body": "Python gives you the object-oriented vocabulary you already use, with roughly a third of the ceremony. It pays for that by moving four guarantees out of the compiler and into your discipline.", "points": ["Immutability is a convention, not a memory layout.", "Interface conformance is structural, checked by shape.", "Numeric exactness is opt-in, and the default is wrong for money.", "Thread safety is not offered at all — there is a global interpreter lock instead."], "aside": "Everything in this episode is a consequence of one of these four."}
```

Here is the one sentence version of the entire topic. Python gives you the same object oriented vocabulary you already use, with about a third of the ceremony. And it pays for that by moving four guarantees out of the compiler and into your discipline. Immutability becomes a convention rather than a memory layout. Interface conformance becomes structural, checked by shape rather than by declaration. Numeric exactness becomes opt in, and the default is the wrong one for money. And thread safety is simply not offered. There is a global interpreter lock instead. Every gotcha in this episode is a consequence of one of those four. Keep them in mind and the rest is derivation.

## s4 · CodeWalkthrough

```props
{"eyebrow": "value objects", "title": "A record becomes a frozen dataclass", "filename": "phases/p0/p0-1-python-for-dotnet/src/p0_1_oms/money.py", "code": "@dataclass(frozen=True, order=True)\nclass Money:\n    amount: Decimal\n    currency: str = \"INR\"\n\n    def __init__(self, amount: Numeric, currency: str = \"INR\") -> None:\n        raw = _to_decimal(amount, \"amount\")\n        if len(currency) != 3 or not currency.isalpha():\n            raise ValidationError(f\"bad currency {currency!r}\")\n        object.__setattr__(self, \"amount\", raw.quantize(_QUANTUM, ROUND_HALF_UP))\n        object.__setattr__(self, \"currency\", currency.upper())", "highlights": [{"at": 30, "lines": [1], "caption": "frozen=True gives value equality, a hash and blocked rebinding — a readonly record struct."}, {"at": 150, "lines": [3, 4], "caption": "Fields are declared, not written out with backing properties."}, {"at": 260, "lines": [7, 8, 9], "caption": "Guard clauses live in __init__ or __post_init__ — the constructor validation you already write."}, {"at": 400, "lines": [10, 11], "caption": "object.__setattr__ is how a frozen dataclass assigns during construction."}]}
```

Here is the money value object. Read it as a record. The decorator, frozen equals true, gives you structural equality, a working hash, a useful string representation, and blocked attribute rebinding. That is very close to a readonly record struct. The fields are declared with type annotations, and there are no backing properties to write. The guard clauses go in the constructor, or in a method called post init, and they do the job that argument validation does in your C sharp constructors. The last two lines look strange. Object dot set attribute. That is the escape hatch a frozen dataclass gives you for assigning fields during construction, because normal assignment is what frozen is blocking. Now here is the first gap. Frozen is shallow. It protects the reference, not what the reference points at. A frozen object holding a list will let you append to that list all day long. Your readonly keyword has exactly the same hole. You notice it less in C sharp because record structs copy.

## s5 · Callout

```props
{"kind": "gotcha", "heading": "The mutable default has no C# analogue", "body": "A default argument is evaluated once, when the function is defined — not on each call. Every call then shares the same object. This is why dataclasses refuse a mutable default outright and make you write field(default_factory=list).", "code": "def add(item, bucket=[]):   # evaluated ONCE\n    bucket.append(item)\n    return bucket\n\nadd(1)   # [1]\nadd(2)   # [1, 2]   <- same list"}
```

This one has no C sharp analogue at all, and it catches everybody. A default argument in Python is evaluated once, at the moment the function is defined. Not on each call. So every call shares the same object. Look at the code. Add one returns a list containing one. Add two returns a list containing one and two. It is the same list. That is why dataclasses refuse a mutable default outright, and make you write field, default factory, list. In a plain function, nothing stops you. The rule to internalise is simple. If a default value is mutable, it must be constructed inside the function, never in the signature.

## s6 · MathReveal

```props
{"eyebrow": "the domain", "title": "Weighted average cost", "english": "A position's cost basis is the total money spent divided by the total shares held — re-weighted every time you add to it.", "equation": "avg' = (avg · |q| + px · |Δ|) / |q + Δ|", "code": "gross = self.average_cost.amount * abs(old) + px * abs(signed)\navg = gross / Decimal(abs(new))", "note": "Reducing a position does not move the basis; only adding does.", "stageFrames": [10, 130, 260]}
```

Now the one piece of arithmetic in this topic, and we will show it three times. First in plain English. A position's cost basis is the total money spent divided by the total shares held, re weighted every time you add to it. Second, in symbols. The new average equals the old average times the absolute old quantity, plus the fill price times the absolute fill quantity, all divided by the absolute new quantity. Notice that every quantity is an absolute value, because this formula has to work for a short position too. And third, in code. Two lines. Gross is the old average times the old size, plus the price times the fill size. The new average is that gross divided by the new size. One more rule that is not in the formula. Reducing a position does not move the basis. Only adding does. When you sell, you realise profit against the existing average and leave the average alone.

## s7 · CodeWalkthrough

```props
{"eyebrow": "the domain", "title": "One fill, four branches", "filename": "src/p0_1_oms/oms.py \u00b7 Position.apply", "code": "signed = side.sign * fill.quantity.shares\nold, px = self.net_quantity, fill.price.amount\nnew = old + signed\nif old == 0 or (old > 0) == (signed > 0):      # opening or adding\n    gross = self.average_cost.amount * abs(old) + px * abs(signed)\n    return replace(self, net_quantity=new, average_cost=gross / abs(new))\nclosed = min(abs(old), abs(signed))            # reducing or flipping\nrealised = (px - self.average_cost.amount) * closed * (1 if old > 0 else -1)\nif new == 0:                                   # flat\n    return replace(self, net_quantity=0, average_cost=ZERO, realised_pnl=pnl)\nif (new > 0) == (old > 0):                     # partial reduction\n    return replace(self, net_quantity=new, realised_pnl=pnl)\nreturn replace(self, net_quantity=new, average_cost=px, realised_pnl=pnl)", "highlights": [{"at": 10, "lines": [1, 2, 3], "caption": "Direction is encoded once, as a sign on the enum. Nothing else in the file knows about buy or sell."}, {"at": 40, "lines": [4, 5, 6], "caption": "Opening or adding: re-weight the average, realise nothing."}, {"at": 70, "lines": [7, 8], "caption": "Reducing: realise against the existing basis, on the shares actually closed."}, {"at": 95, "lines": [9, 10], "caption": "Flat: the basis goes back to zero."}, {"at": 115, "lines": [11, 12], "caption": "Partial reduction: the basis is untouched."}, {"at": 135, "lines": [13], "caption": "Flipped through zero: the remainder opens at the fill price."}]}
```

Here is the whole of position dot apply, and it has exactly four branches. Start at the top. Signed is the fill quantity multiplied by the side's sign, which is plus one for a buy and minus one for a sell. That sign lives on the enum, and it is the only place in the file that knows the difference between buying and selling. Everything downstream is arithmetic. Branch one. If we were flat, or if the fill pushes in the same direction we already hold, we are opening or adding. Re weight the average, realise nothing. Branch two. Otherwise we are reducing, and we realise profit on the number of shares actually closed, against the existing average cost. Branch three. If the new quantity is zero, we are flat, and the basis goes back to zero. Branch four. If we are still on the same side, this was a partial reduction, and the basis is untouched. And the final line is the case people forget. We flipped through zero. We sold more than we held. The realised profit covers only the shares that were closed, and the remainder opens a short position at the fill price. Every one of those four branches has a test. That is not thoroughness for its own sake. It is that three of them are indistinguishable from each other until the day a position flips.

## s8 · ChartScene

```props
{"eyebrow": "verification", "title": "Where the storage precision came from", "kind": "bar", "bars": [{"label": "4 dp storage", "value": 2.77e-05, "color": "#e05e6b", "note": "28x over tolerance"}, {"label": "tolerance", "value": 1e-06, "color": "#e0b04e", "note": "the requirement"}, {"label": "6 dp storage", "value": 3.11e-07, "color": "#5ac48c", "note": "shipped"}], "logScale": true, "caption": "Worst relative error replaying 210 orders, against an independently written float reducer."}
```

Now the part that made this topic worth building. The verification benchmark replays two hundred and ten synthetic orders through the domain model, and reconciles the result against two references that share no code with it. One is a pandas group by of signed quantities. The other is an average cost reducer written out separately in plain floating point. The quantity reconciliation passed immediately. The profit and loss reconciliation did not. The worst relative error was two point eight times ten to the minus five, which is twenty eight times the tolerance. Nothing was wrong. An average cost is a quotient, and rounding it to four decimal places on every single fill accumulates drift over a few hundred orders. Storing at six decimal places and displaying at four dropped the error to three point one times ten to the minus seven. That is the whole lesson of the topic compressed into one number. Precision is a property of the operation, not of the type. Choosing decimal was necessary and it was not sufficient.

## s9 · Callout

```props
{"kind": "gotcha", "heading": "Return NotImplemented, do not raise", "body": "When a binary operator does not recognise the other operand, return the NotImplemented singleton. Python then tries the reflected method on the right-hand side, and only raises TypeError if that fails too. Raising NotImplementedError instead permanently breaks other + self, and blames the wrong class in the traceback.", "code": "def __add__(self, other):\n    if not isinstance(other, Money):\n        return NotImplemented    # not raise!\n    return Money(self.amount + other.amount)"}
```

Operator overloading translates almost directly, but there is one line in it that surprises everybody. When your addition method does not recognise the operand on the right, you return the not implemented singleton. You do not raise. Returning it tells Python, I do not know how to handle this, try the other operand. Python then calls the reflected method on the right hand side, and only raises a type error if that also fails. If you raise not implemented error instead, you permanently break the expression something else plus money, and the traceback blames the wrong class. C sharp has no equivalent because operator resolution there is static and happens at compile time.

## s10 · ConceptScene

```props
{"eyebrow": "the disappearing interface", "title": "Enums carry behaviour. Interfaces evaporate.", "body": "A C# enum is a named integer, and behaviour arrives via extension methods. A Python enum member is an object, so Side.BUY.sign lives where it belongs. The interface goes further: it disappears entirely.", "points": ["class NoShortSelling: def check(self, order) -> None: ...", "It never names RiskCheck. It satisfies it anyway.", "Protocols are structural — conformance is by shape, checked statically or at runtime.", "The instinct to declare the interface first is the one to unlearn."], "aside": "typing.Protocol · runtime_checkable · static duck typing"}
```

Two more translations, and the second one is a genuine shift in how you design. An enum in C sharp is a named integer, and you attach behaviour to it with extension methods. A Python enum member is a real object, so side dot buy dot sign, and side dot buy dot opposite, live directly on the enum where they belong. Now the bigger one. The interface disappears. Look at the class on screen. It is called no short selling, it has a method called check that takes an order, and it never mentions any interface anywhere. It satisfies the risk check protocol regardless, because protocols are structural. Conformance is by shape. Your instinct, coming from dot NET, will be to define the interface first and then implement it. That is the instinct to unlearn. Write the concrete class. Add the protocol only where a function needs to state, in its signature, what shape of thing it accepts.

## s11 · Callout

```props
{"kind": "warning", "heading": "The truthiness bug that shipped on day one", "body": "Empty collections, zero, Money(0) and an empty registry are all falsy. So `x = arg or DEFAULT` silently replaces any falsy argument, not just None. This exact line routed every component registration to the global registry whenever a caller passed a fresh, empty one — because Registry.__len__ returns zero.", "code": "target = registry or REGISTRY          # bug\ntarget = REGISTRY if registry is None else registry   # fix"}
```

And here is a real bug from this repository, written on the first day, caught by the first test that used it. The AlphaDesk component registry has a line that chooses which registry to file into. Registry or the global registry. It reads as obviously correct. It is not. An empty registry defines a length of zero, which makes it falsy, so any caller passing a fresh empty registry silently got the global one instead. The test failed with, assert zero equals one, which is a wonderfully unhelpful message until you see it. The fix is the second line. Is none, never or, for optional arguments. Write that one on a sticky note. In a language where zero, empty string, empty list and a zero valued money object are all falsy, the or idiom is a trap with a very wide mouth.

## s12 · ConceptScene

```props
{"eyebrow": "async", "title": "Same keywords, different machine", "body": "async and await look identical to yours and are not the same thing underneath.", "points": ["Coroutines are cold. Calling an async def runs nothing until you await it.", "One thread, one event loop. No thread pool. async is concurrency over I/O, never parallelism over CPU.", "No ConfigureAwait, no synchronisation context, and therefore none of the classic deadlocks — but you cannot block on a coroutine from sync code at all.", "Task.WhenAll is asyncio.gather. CancellationToken is a CancelledError raised at the next await point."], "aside": "asyncio.run() is the only door in, and it refuses to nest."}
```

Last translation, and it is the one that looks most familiar and behaves least like you expect. Async and await are spelled the same and they run on a different machine. Four differences, in the order they will hurt you. First, coroutines are cold. In dot NET, a task returned from a method is usually already running before you await it. In Python, calling an async function returns an object that has done nothing at all. Forget the await, and the work simply never happens. Second, there is one thread and one event loop. There is no thread pool underneath. Async buys you concurrency over input and output, and never parallelism over computation. A tight numeric loop inside a coroutine starves every other task on the loop. Third, there is no configure await, no synchronisation context, and therefore none of the classic deadlocks you have learned to avoid. The price is that you cannot block on a coroutine from synchronous code at all. Async io dot run is the only door in, and it refuses to nest. And fourth, task dot when all is asyncio dot gather, and a cancellation token becomes an exception raised at the next await point inside the task.

## s13 · DiagramScene

```props
{"eyebrow": "the lifecycle", "title": "One order, five states", "nodes": [{"id": "new", "label": "NEW", "sub": "placed, unfilled", "x": 0.13, "y": 0.3, "w": 0.19}, {"id": "partial", "label": "PARTIAL", "sub": "some leaves", "x": 0.45, "y": 0.3, "w": 0.19}, {"id": "filled", "label": "FILLED", "sub": "terminal", "x": 0.79, "y": 0.3, "w": 0.19, "color": "#5ac48c"}, {"id": "cancelled", "label": "CANCELLED", "sub": "terminal", "x": 0.45, "y": 0.78, "w": 0.21, "color": "#e08a4e"}, {"id": "rejected", "label": "REJECTED", "sub": "terminal · risk", "x": 0.13, "y": 0.78, "w": 0.21, "color": "#e05e6b"}], "edges": [{"from": "new", "to": "partial", "label": "fill", "appearAt": 40}, {"from": "partial", "to": "filled", "label": "last fill", "appearAt": 70}, {"from": "new", "to": "rejected", "label": "risk check", "appearAt": 100}, {"from": "new", "to": "cancelled", "label": "cancel", "appearAt": 130}, {"from": "partial", "to": "cancelled", "label": "cancel", "appearAt": 150}], "caption": "Three terminal states. A partially filled order can be cancelled but never rejected."}
```

Put it together and you get the order lifecycle. Five states, three of them terminal. A new order takes a fill and becomes partial. It takes its last fill and becomes filled. A pre trade risk check can reject it before any fill. And it can be cancelled from new or from partial. There is one asymmetry on this diagram worth pausing on. A partially filled order can be cancelled, but it can never be rejected. Rejection means the order should never have existed. Once shares have traded, that statement is no longer available to you. This is the kind of invariant that is obvious when stated and completely invisible in code that does not enforce it, which is why there is a test named exactly that.

## s14 · RecapScene

```props
{"eyebrow": "P0.1", "title": "Recap", "points": ["Dataclasses give you records; frozen is shallow, and a mutable default is shared across calls.", "Money is Decimal — and the precision of the operation, not of the type, is what the benchmark measures.", "Protocols replace interfaces structurally; is None, never or, for optional arguments.", "Coroutines are cold, single-threaded, and cancelled by an exception at the next await point."], "ifSkipped": "Phase 1's autograd engine is operator overloading plus a graph walk. Without today, __add__ returning NotImplemented and a gradient of 0.0 being falsy will read as \"autograd is confusing\".", "next": "P0.2 · NumPy as your new LINQ"}
```

Four things to take away. Dataclasses give you records, frozen is shallow, and a mutable default argument is shared across every call. Money is decimal, and the benchmark measures the precision of the operation rather than the precision of the type. Protocols replace interfaces structurally, and you should write is none, never or, for optional arguments. And coroutines are cold, single threaded, and cancelled by an exception delivered at the next await point. Now, what breaks if we skip this. Phase one builds an autograd engine, and its entire design is operator overloading plus a graph walk. Without today's episode, an add method returning not implemented, and a gradient of zero point zero being falsy, will read as autograd is confusing. When what is actually happening is that Python is doing exactly what it told you it would. Next episode. NumPy as your new LINQ.
