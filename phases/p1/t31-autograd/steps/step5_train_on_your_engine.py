#!/usr/bin/env python3
"""Step 5 — train the Phase-0 task on your engine and prove the curve is right.

Run:  python3 steps/step5_train_on_your_engine.py

The "done when" for this topic is not accuracy. It is that a network trained by
hand-written backward closures follows the *same loss curve* as one trained by
gradients derived independently on paper. Two implementations, one trajectory.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t31_autograd.train import make_features, train_reference, train_with_engine


def main() -> None:
    X, y = make_features(n=512, d=8, seed=0)
    engine, model = train_with_engine(X, y, steps=200)
    reference = train_reference(X, y, steps=200)

    a, b = np.array(engine.losses), np.array(reference.losses)
    print(f"  parameters in the model: {model.num_parameters()}")
    print(f"  step   0: engine {a[0]:.6f}   reference {b[0]:.6f}")
    print(f"  step  50: engine {a[50]:.6f}   reference {b[50]:.6f}")
    print(f"  step 199: engine {a[-1]:.6f}   reference {b[-1]:.6f}")
    print(f"\n  max divergence over 200 steps: {np.abs(a - b).max():.3e}")
    print(f"  training accuracy: engine {engine.accuracy:.3f}, reference {reference.accuracy:.3f}")
    print("\n  Reminder from P0.3, still true: fitting a toy task is not a signal.")
    assert np.abs(a - b).max() < 1e-9


if __name__ == "__main__":
    main()
