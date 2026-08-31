---
topic: t31-e1
episode: 1
title: The chain rule, turned into code
voice: am_michael
speed: 0.85
runtime_target_minutes: 11
paper: Karpathy micrograd lineage; CS231n backpropagation notes
---

## s1 · TitleCard

```props
{"title": "Autograd, part one", "subtitle": "The chain rule is three lines of bookkeeping per operation. Here they are.", "topicId": "T31", "episode": "Episode 1"}
```

Welcome back to the AI Systems Forge. This is phase one, topic thirty one. The autograd engine. In phase zero you called dot backward and a number appeared. This topic removes the magic. By the end of these three episodes you will have written reverse mode automatic differentiation twice, once on scalars and once on arrays, and you will never again wonder what a framework is doing when you call that method.

## s2 · ArchitectureMap

```props
{"eyebrow": "where we are", "highlight": ["training"], "caption": "T31 rebuilds the training loop's differentiation on your own engine."}
```

Here is where we are. AlphaDesk is the fictional trading desk this course builds, one topic at a time. The lit block is the training loop. Phase zero borrowed PyTorch's autograd to fill it. This topic replaces the borrowed part with your own. Nothing else on the desk changes, which is the point. Once you know what dot backward does, every later topic that trains something is one less thing you are taking on faith.

## s3 · ConceptScene

```props
{"eyebrow": "the premise", "title": "An expression is a graph, and derivatives flow backwards through it", "body": "Evaluating an expression forwards is ordinary arithmetic. What is surprising is that every partial derivative of the final value, with respect to every input, can be had in one backwards sweep — for roughly the cost of the forward pass.", "points": ["Each node knows one local fact: how its output moves when its inputs move.", "The chain rule glues those local facts into a global answer.", "No node ever knows what the loss is. That is what makes it composable.", "Reverse mode is cheap when there is one output and many inputs — which is exactly a loss function."], "aside": "Forward mode is the other way round: cheap for one input and many outputs. Neural networks are the other shape, so reverse mode won."}
```

Here is the whole idea before any code. A numeric expression is a directed graph. The leaves are your inputs, the interior nodes are operations, and the root is the number you care about. Evaluating it forwards is ordinary arithmetic, and nobody is impressed. What is surprising is the backwards direction. Every partial derivative of the root, with respect to every leaf, falls out of a single sweep in reverse, for roughly the cost of the forward pass. And it works because each node only ever needs to know one local fact. How does my output move when my inputs move. Multiplication knows that. Tanh knows that. Neither of them knows what a loss function is, and that ignorance is exactly what lets you compose them without limit. One aside worth banking. This is reverse mode. It is cheap when you have one output and many inputs. There is a forward mode which is the mirror image, cheap for one input and many outputs. A neural network has one loss and millions of parameters, so reverse mode is the one that won.

## s4 · DiagramScene

```props
{"eyebrow": "the mechanism", "title": "One node, two directions", "nodes": [{"id": "a", "label": "a", "sub": "leaf", "x": 0.12, "y": 0.3, "appearAt": 8}, {"id": "b", "label": "b", "sub": "leaf", "x": 0.12, "y": 0.72, "appearAt": 14}, {"id": "m", "label": "a x b", "sub": "op node", "x": 0.44, "y": 0.5, "appearAt": 30, "color": "#5ac48c"}, {"id": "t", "label": "tanh", "sub": "op node", "x": 0.74, "y": 0.5, "appearAt": 48}, {"id": "L", "label": "L", "sub": "the root", "x": 0.94, "y": 0.5, "w": 0.1, "appearAt": 60}], "edges": [{"from": "a", "to": "m", "appearAt": 34}, {"from": "b", "to": "m", "appearAt": 38}, {"from": "m", "to": "t", "appearAt": 52}, {"from": "t", "to": "L", "appearAt": 64}, {"from": "L", "to": "t", "label": "grad", "appearAt": 90, "dashed": true}, {"from": "t", "to": "m", "label": "grad", "appearAt": 100, "dashed": true}, {"from": "m", "to": "a", "label": "b x grad", "appearAt": 112, "dashed": true}], "caption": "Solid arrows are the forward pass. Dashed arrows are the same graph, walked in reverse."}
```

Look at the smallest interesting graph. Two leaves, a and b. They feed a multiply. The multiply feeds a tanh. The tanh is the root, and we will call it L. The solid arrows are the forward pass, and they are just evaluation. Now the dashed arrows. They are the same graph walked backwards. The root starts with a gradient of one, because the derivative of anything with respect to itself is one. Tanh receives that and passes on one minus tanh squared. The multiply receives that and hands each of its inputs the upstream gradient scaled by the other input. That is the entire mechanism. Every dashed arrow is one small multiplication that the node it belongs to already knew how to do.

## s5 · CodeWalkthrough

```props
{"eyebrow": "the code", "title": "A scalar that remembers where it came from", "filename": "phases/p1/t31-autograd/src/t31_autograd/engine.py", "code": "class Value:\n    def __init__(self, data, _children=(), _op=\"\"):\n        self.data = float(data)\n        self.grad = 0.0\n        self._backward = lambda: None\n        self._prev = tuple(_children)\n\n    def __mul__(self, other):\n        other = self._coerce(other)\n        out = Value(self.data * other.data, (self, other), \"*\")\n\n        def _backward():\n            self.grad += other.data * out.grad\n            other.grad += self.data * out.grad\n\n        out._backward = _backward\n        return out", "highlights": [{"at": 20, "lines": [3, 4], "caption": "Two numbers per node: the forward value, and the slot the gradient will land in."}, {"at": 110, "lines": [5, 6], "caption": "A closure for the local rule, and the children — that is the graph, stored implicitly."}, {"at": 210, "lines": [10], "caption": "The forward value is computed immediately; the node records who made it."}, {"at": 300, "lines": [12, 13, 14], "caption": "The local rule. d(ab)/da = b, so each input gets the upstream gradient scaled by the OTHER input."}, {"at": 400, "lines": [13, 14], "caption": "Plus-equals, not equals. This single character is the most common bug in a hand-rolled engine."}]}
```

Here is the class. Look at how little there is. A node holds two numbers. Data, which is the forward value, and grad, which is the slot the derivative will land in. Then it holds a closure called backward, and a tuple of children. Those two fields are the graph. Nobody builds a separate graph object. It exists because every node remembers the nodes it was made from. Now the multiply. It computes the forward value straight away, and constructs the output node with itself and the other operand as children. Then it defines the local rule and attaches it. Read the two lines inside that closure. The derivative of a times b, with respect to a, is b. So a receives the upstream gradient scaled by b. And symmetrically for b. That is the whole of the chain rule for multiplication, and it fits on two lines. And now look very carefully at the operator on those two lines. It is plus equals. Not equals. That single character is the most common bug in a hand written autograd engine, and it is what the next scene is about.

## s6 · Callout

```props
{"kind": "gotcha", "heading": "Accumulate, never assign", "body": "If a node feeds two consumers, it receives gradient from both, and the total is the sum. Write = instead of += and you silently keep only the last contribution — the network still trains, slightly worse, forever.", "code": "x = Value(3.0)\ny = x * x + x        # x is used twice\ny.backward()\n\nx.grad == 7.0        # 2x + 1, correct\n# with '=' instead of '+=' you get 6.0 or 1.0,\n# depending on which parent ran last."}
```

Here is why. Consider y equals x times x, plus x. The node x is used twice, so on the way back it receives gradient from two different parents, and the correct answer is the sum of the two. Two x plus one, which at x equals three is seven. Now suppose you had written plain assignment. The last parent to run overwrites whatever the first one deposited, and you get six, or you get one, depending on evaluation order. Notice what does not happen. Nothing raises. No shape check fails. The loss still goes down, just a little worse than it should, forever. That is the character of autograd bugs, and it is why the last scene of this episode is about how you catch them.

## s7 · MathReveal

```props
{"eyebrow": "the rule", "title": "The chain rule, stated as an algorithm", "english": "The derivative of the root with respect to a node is the sum, over every consumer of that node, of the consumer's own gradient times the local derivative between them.", "equation": "dL/dv = SUM over consumers c of ( dL/dc x dc/dv )", "code": "def _backward():\n    self.grad += LOCAL_DERIVATIVE * out.grad", "note": "The sum over consumers is the '+=' from the previous scene. The local derivative is the only thing each operator has to supply.", "stageFrames": [10, 140, 280]}
```

Let us say the rule three times, because it is the only piece of mathematics in this episode. First in English. The derivative of the root with respect to some node is the sum, over every consumer of that node, of the consumer's own gradient multiplied by the local derivative between them. Second, in symbols. d L by d v equals the sum over consumers c of d L by d c, times d c by d v. And third, in code, where it collapses to a single line. Self dot grad plus equals the local derivative times out dot grad. Look at how the three map onto each other. The sum over consumers is the plus equals. The consumer's own gradient is out dot grad, which was filled in earlier in the sweep. And the local derivative is the only thing any individual operator has to supply. Add, multiply, tanh, exponent, logarithm. Five lines of calculus you already know, and the engine is done.

## s8 · CodeWalkthrough

```props
{"eyebrow": "the operator set", "title": "Five local rules and everything else is sugar", "filename": "src/t31_autograd/engine.py", "code": "def exp(self):\n    e = math.exp(self.data)\n    out = Value(e, (self,), \"exp\")\n    out._backward = lambda: setattr(self, 'grad', self.grad + e * out.grad)\n    return out\n\ndef tanh(self):\n    t = math.tanh(self.data)\n    ...  self.grad += (1 - t * t) * out.grad\n\ndef relu(self):\n    ...  self.grad += (out.data > 0) * out.grad\n\ndef __neg__(self):      return self * -1\ndef __sub__(self, o):   return self + (-o)\ndef __truediv__(self, o): return self * (o ** -1)", "highlights": [{"at": 20, "lines": [1, 2, 3, 4], "caption": "exp is its own derivative — and we reuse the forward value rather than recomputing it."}, {"at": 130, "lines": [7, 8, 9], "caption": "tanh caches its forward value too: 1 - t^2 needs the output, not the input."}, {"at": 240, "lines": [11, 12], "caption": "ReLU has no derivative at exactly zero. Every framework picks a subgradient; we pick 0, matching PyTorch."}, {"at": 340, "lines": [14, 15, 16], "caption": "Subtraction and division are not primitives. Fewer primitives, fewer backward rules to get wrong."}]}
```

Here is the rest of the operator set, and there is less of it than you expect. Exponent is the pleasant one. It is its own derivative, and notice we reuse the forward value we already computed rather than calling exp twice. Tanh does the same trick, because one minus t squared needs the output, not the input. ReLU is the interesting one. At exactly zero it has no derivative at all. The function has a corner. Every framework picks a subgradient there, and the choice is arbitrary. We pick zero, which is what PyTorch does. The sin is not picking a convention, it is failing to write down which one you picked. And then look at the bottom three lines. Subtraction is not a primitive. It is addition of a negation. Division is not a primitive. It is multiplication by a reciprocal. Every primitive you add is another backward rule you can get wrong, so the right instinct in an autograd engine is to have as few of them as you can live with.

## s9 · ConceptScene

```props
{"eyebrow": "the defence", "title": "Gradcheck, or you are guessing", "body": "An autodiff bug does not raise. It returns a number of the right sign and roughly the right magnitude, the loss still falls, and you lose a week. The only defence is comparing against a derivative computed a completely different way.", "points": ["Central differences: (f(x+h) - f(x-h)) / 2h, error of order h squared.", "Forward differences are error of order h — about three fewer correct digits, for the same effort.", "Compare relatively: |a - b| / max(eps, |a| + |b|), so the test does not care about scale.", "Run it on random graphs, not on the one example you had in mind."], "aside": "On thirty random scalar graphs the worst relative error is 3.6e-10. That number is the reason to trust the next two episodes."}
```

And now the part that makes the difference between an engine and a plausible looking engine. Gradcheck. The problem with an autodiff bug is that it does not raise. It hands back a number of the right sign and roughly the right magnitude, the loss still goes down, and you lose a week. So the only real defence is to compute the same derivative a completely different way and compare. Finite differences. Take the function at x plus h, subtract the function at x minus h, divide by two h. That is the central difference, and its error goes like h squared. The one sided version, f of x plus h minus f of x, over h, has error going like h, which costs you about three correct digits for exactly the same work. Compare the two answers relatively, not absolutely, so that the test does not care whether your gradients are of size one or size ten thousand. And run it on randomly generated expressions, not on the one example you had in your head while writing the code, because that example is the one your bug already agrees with. On thirty random scalar graphs, this engine's worst relative error is three point six times ten to the minus ten. That number is the reason you can believe the next two episodes.

## s10 · RecapScene

```props
{"eyebrow": "episode one", "points": ["A node stores a value, a gradient slot, its children, and one closure for its local derivative.", "The chain rule is: accumulate, over every consumer, the consumer's gradient times the local derivative.", "Plus-equals, not equals — a node used twice receives gradient twice.", "Keep the primitive set small; subtraction and division are sugar over negation and reciprocal.", "Gradcheck with central differences, on random graphs, or you are guessing."], "ifSkipped": "Skip this and every later topic's '.backward()' stays magic — including Flash Attention in Phase 7, which is nothing but this backward pass written to fit in cache.", "next": "Episode 2: the backward pass, in the only order that is correct."}
```

To recap. A node stores four things. A value, a gradient slot, its children, and one closure holding its local derivative. The chain rule, as an algorithm, says to accumulate over every consumer the consumer's gradient times the local derivative. Plus equals, never equals, because a node used twice receives gradient twice. Keep the primitive set small, since subtraction and division are sugar over negation and reciprocal. And gradcheck with central differences on random graphs, because an autodiff bug will not announce itself. If you skip this topic, every later dot backward in the course stays magic, including Flash Attention in phase seven, which is nothing more than this backward pass rewritten to fit in cache. Next episode. We have all the local rules. Now we have to run them in the right order, and there is exactly one right order.
