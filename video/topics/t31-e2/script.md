---
topic: t31-e2
episode: 2
title: The backward pass, line by line
voice: am_michael
speed: 0.85
runtime_target_minutes: 10
paper: CS231n backpropagation notes
---

## s1 · TitleCard

```props
{"title": "Autograd, part two", "subtitle": "There is exactly one correct order to run the local rules in. Here is why, and what going wrong looks like.", "topicId": "T31", "episode": "Episode 2"}
```

Episode two of the autograd engine. Last time we built the local rules, one per operation. Today we run them, and the entire content of this episode is the word order. Because the local rules are individually correct and collectively useless until they execute in the right sequence.

## s2 · ConceptScene

```props
{"eyebrow": "the constraint", "title": "A node may only push once all of its consumers have pushed into it", "body": "Node v hands gradient to its children. But v's own gradient is a sum over its consumers, and it is only complete once every consumer has contributed. Run v too early and it pushes a partial sum downstream, and everything below inherits the error.", "points": ["Correct order: every node strictly after all of its consumers.", "That is a topological sort of the graph, reversed.", "Insertion order is not it. Depth is not it. Recursion order is not it.", "Get it wrong and nothing raises — you get a plausible, smaller number."], "aside": "This is the same constraint a build system has: compile a dependency before its dependents. Reverse mode is 'make', run backwards."}
```

Here is the constraint, and it is the only thing you have to hold onto. A node hands gradient to its children. But that node's own gradient is a sum over all of its consumers, and the sum is only complete once every consumer has contributed. So if you run a node too early, it pushes a partial sum downstream, and every node below it inherits the mistake. The correct order is therefore. Every node strictly after all of its consumers. Which is a topological sort of the graph, reversed. And it is worth naming three orders that look plausible and are not it. The order you created the nodes in is not it. Sorting by depth is not it. And the order that naive recursion happens to visit in is not it either, though it comes closest. If you have ever written a build system, this is the same constraint. Compile a dependency before the things that depend on it. Reverse mode autodiff is make, run backwards.

## s3 · DiagramScene

```props
{"eyebrow": "the failure", "title": "The diamond: one node, two paths", "nodes": [{"id": "x", "label": "x", "sub": "used twice", "x": 0.12, "y": 0.5, "color": "#e0b04e", "appearAt": 6}, {"id": "a", "label": "a = 2x", "x": 0.44, "y": 0.24, "appearAt": 20}, {"id": "b", "label": "b = tanh(x)", "x": 0.44, "y": 0.76, "appearAt": 30}, {"id": "y", "label": "y = a x b", "x": 0.8, "y": 0.5, "appearAt": 44, "color": "#5ac48c"}], "edges": [{"from": "x", "to": "a", "appearAt": 24}, {"from": "x", "to": "b", "appearAt": 34}, {"from": "a", "to": "y", "appearAt": 48}, {"from": "b", "to": "y", "appearAt": 52}, {"from": "y", "to": "a", "label": "1st", "appearAt": 80, "dashed": true}, {"from": "y", "to": "b", "label": "2nd", "appearAt": 90, "dashed": true}, {"from": "a", "to": "x", "label": "3rd", "appearAt": 100, "dashed": true}, {"from": "b", "to": "x", "label": "4th", "appearAt": 110, "dashed": true}], "caption": "x must run last. Run it after 'a' but before 'b' and you keep only half its gradient."}
```

The smallest graph that can catch you is the diamond. x feeds two different operations. a is two x. b is tanh of x. And y multiplies them together. Watch the dashed arrows and their numbers. y runs first. It pushes into a, then into b. Then a pushes into x. Then b pushes into x. And only now is x's gradient complete. Now imagine we ran x after a but before b. x would push a gradient downstream that is missing the entire tanh path. In this tiny graph x is a leaf, so nothing is downstream and the damage is contained to a wrong final answer. In a real network x is a hidden activation with fifty layers under it, and every one of them is now wrong by a factor nobody can see.

## s4 · CodeWalkthrough

```props
{"eyebrow": "the code", "title": "An iterative post-order traversal", "filename": "src/t31_autograd/engine.py", "code": "def topo_order(self):\n    order, seen = [], set()\n    stack = [(self, False)]\n    while stack:\n        node, expanded = stack.pop()\n        if expanded:\n            order.append(node)\n            continue\n        if id(node) in seen:\n            continue\n        seen.add(id(node))\n        stack.append((node, True))\n        for child in node._prev:\n            if id(child) not in seen:\n                stack.append((child, False))\n    return order", "highlights": [{"at": 20, "lines": [3], "caption": "The flag says whether this node's children have already been pushed."}, {"at": 130, "lines": [5, 6, 7], "caption": "Second visit: children are done, so the node is safe to emit."}, {"at": 230, "lines": [9, 10, 11], "caption": "First visit: mark it seen, so a diamond does not enqueue it twice."}, {"at": 320, "lines": [12, 13, 14, 15], "caption": "Push self back with the flag set, THEN the children — the stack is LIFO, so children pop first."}]}
```

Here is the traversal, and it is worth reading slowly because the two phase trick is not obvious the first time. Each stack entry is a node plus a boolean saying whether its children have already been pushed. Pop an entry. If the flag is set, the children are already emitted, so this node is safe to append. If the flag is clear, this is the first time we have reached the node. Mark it seen, so a diamond cannot enqueue it twice. Then push the node back with the flag set, and after that push its children. The stack is last in first out, so the children pop before the node does, which is exactly the post order we want. The result is a list where every node appears after all of its children. Reverse that list and every node appears after all of its consumers, which is the constraint from two scenes ago.

## s5 · Callout

```props
{"kind": "warning", "heading": "Why this is iterative and not recursive", "body": "The obvious implementation is a recursive depth-first walk, and it is correct. It also dies. Unroll one small network to scalars and the graph is tens of thousands of nodes deep; Python's default recursion limit is one thousand.", "code": "v = Value(1.0)\nfor _ in range(20_000):\n    v = v * 1.0001\nv.backward()      # recursive: RecursionError\n                  # iterative: fine, and there is a test for it"}
```

And here is why we did it with an explicit stack rather than recursion. The recursive version is three lines and it is correct, and it dies. Take a single value and multiply it by a constant twenty thousand times. That is a chain twenty thousand nodes deep, which is not an unusual depth once you unroll even a small network to scalars. Python's default recursion limit is one thousand. The recursive implementation raises. The iterative one does not, and there is a test in the suite that does exactly this, because it is the kind of limit you would rather find in continuous integration than during a training run.

## s6 · CodeWalkthrough

```props
{"eyebrow": "the code", "title": "backward(), in full", "filename": "src/t31_autograd/engine.py", "code": "def backward(self):\n    order = self.topo_order()\n    for node in order:\n        node.grad = 0.0\n    self.grad = 1.0\n    for node in reversed(order):\n        node._backward()", "highlights": [{"at": 20, "lines": [2], "caption": "One traversal, reused for both loops."}, {"at": 120, "lines": [3, 4], "caption": "Zero first: calling backward() twice must not double the gradients."}, {"at": 240, "lines": [5], "caption": "The seed. d(self)/d(self) = 1 — the only derivative in the whole system you write by hand."}, {"at": 340, "lines": [6, 7], "caption": "Reverse topological order, one call each. That is the algorithm."}]}
```

And here is the whole backward pass. Seven lines. Compute the order once. Zero every gradient, because calling backward twice on the same graph must not double the numbers. Set the root's gradient to one, which is the seed, and the only derivative in the entire system that you write down by hand. Then walk the order in reverse and call each node's closure exactly once. That is it. Everything else in this episode was explaining why line six says reversed.

## s7 · Callout

```props
{"kind": "insight", "heading": "PyTorch made the opposite choice about zeroing", "body": "This engine zeroes gradients at the start of backward(). PyTorch accumulates across calls and makes you call zero_grad() yourself — which is why forgetting it is the classic beginner bug, and also why gradient accumulation over micro-batches is free.", "code": "# here\nloss.backward()          # gradients are exactly this batch's\n\n# PyTorch\nopt.zero_grad()\nloss.backward()          # or they add to whatever was there"}
```

There is a design decision hiding in line three that is worth pulling out, because you will meet the other side of it constantly. This engine zeroes gradients at the start of backward. PyTorch does not. PyTorch accumulates across calls and makes you call zero grad yourself. That choice is why forgetting zero grad is the single most common beginner bug in PyTorch. It is also why gradient accumulation across micro batches is completely free there. Call backward four times, step once, and you have trained on a batch four times larger than fits in memory. Neither choice is wrong. But an engine has to pick one and say so, because the two behave identically right up until they do not.

## s8 · ChartScene

```props
{"eyebrow": "verification", "title": "The order is not a matter of taste", "kind": "bar", "bars": [{"label": "reverse topological", "value": 2.929, "color": "#5ac48c", "note": "correct"}, {"label": "analytic answer", "value": 2.929, "color": "#7a8aa0", "note": "2·tanh(x) + 2x(1-tanh²x)"}, {"label": "insertion order", "value": 1.995, "color": "#e05e6b", "note": "32% low, no error raised"}], "caption": "d/dx of 2x·tanh(x) at x=1.5, computed three ways. The wrong one is not obviously wrong."}
```

And to make the point concrete, here is the diamond evaluated numerically. The derivative of two x times tanh x, at x equals one point five. Reverse topological order gives two point nine two nine. The analytic answer, worked out on paper, gives two point nine two nine. Insertion order gives one point nine nine five. Look at that third bar for a moment. It is the right sign. It is the right order of magnitude. It would train a network. It is thirty two percent wrong and nothing anywhere raised an exception. This is what I mean when I say autograd bugs do not announce themselves, and it is why the gradcheck from episode one is not optional infrastructure.

## s9 · RecapScene

```props
{"eyebrow": "episode two", "points": ["A node may push gradient only after every one of its consumers has pushed into it.", "That ordering is a topological sort, reversed — not insertion order, not depth.", "Use an explicit two-phase stack: a scalar-unrolled network is tens of thousands of nodes deep.", "backward() = one traversal, zero everything, seed the root with 1.0, run in reverse.", "Zeroing policy is a design choice; PyTorch made the other one, which is why zero_grad() exists."], "ifSkipped": "The wrong order does not crash. It returns a plausible number that is 32% low, and trains a slightly worse model forever.", "next": "Episode 3: the same engine on arrays — and broadcasting, which is where the day goes."}
```

To recap. A node may push gradient only after every one of its consumers has pushed into it. That ordering is a topological sort, reversed, and it is not insertion order and not depth. Implement the traversal with an explicit two phase stack, because a scalar unrolled network is tens of thousands of nodes deep and recursion will not survive it. Backward itself is one traversal, zero everything, seed the root with one, run in reverse. And the zeroing policy is a genuine design choice where PyTorch went the other way, which is the entire reason zero grad exists. Skip this and the wrong order will not crash. It will hand you a plausible number that is a third too small, and train a slightly worse model forever. Next episode is the hard one. The same engine, on arrays, where broadcasting turns a one line convenience in the forward pass into a summation you have to get exactly right on the way back.
