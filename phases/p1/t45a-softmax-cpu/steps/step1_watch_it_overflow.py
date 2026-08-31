#!/usr/bin/env python3
"""Step 1 — write the formula from the textbook and watch it die.

Run:  python3 steps/step1_watch_it_overflow.py

softmax(x)_i = exp(x_i) / sum_j exp(x_j). Every symbol is correct. The problem
is that `exp` has a finite range in every floating-point format, and attention
logits routinely leave it.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t45a_softmax.softmax import naive_softmax, stable_softmax


def where_exp_dies() -> None:
    print("  dtype     max finite   largest x with finite exp(x)")
    for dtype, limit in ((np.float16, 11), (np.float32, 88), (np.float64, 709)):
        info = np.finfo(dtype)
        print(f"  {np.dtype(dtype).name:<9} {float(info.max):<12.3g} {limit}")
    print("\n  float16 gives up at eleven. Not 1e11 -- eleven.")


def the_failure() -> None:
    cases = {
        "ordinary logits": np.array([[1.0, 2.0, 3.0]]),
        "large but plausible": np.array([[90.0, 91.0, 92.0]]),
        "adversarial (+/-1e4)": np.array([[1e4, -1e4, 0.0]]),
        "all equal and large": np.array([[800.0, 800.0, 800.0]]),
    }
    print("\n  naive vs stable, float64:")
    for name, x in cases.items():
        with np.errstate(over="ignore", invalid="ignore"):
            naive = naive_softmax(x)
        stable = stable_softmax(x)
        ok = "ok " if np.allclose(naive, stable, equal_nan=False) else "BROKEN"
        print(f"    {name:<22} naive={np.array2string(naive[0], precision=4)!s:<32} {ok}")
        print(f"    {'':<22} stable={np.array2string(stable[0], precision=4)}")


def underflow_is_the_quiet_one() -> None:
    """Overflow shouts (inf/nan). Underflow whispers."""
    x = np.array([[0.0, -800.0]])
    p = stable_softmax(x)
    print("\n  underflow: softmax([0, -800]) =", p[0])
    print("  the second probability is exactly 0, not 1e-348 -- it fell below")
    print("  the smallest subnormal. That is *correct to the dtype*, and it is")
    print("  also why log(softmax(x)) is -inf and why log_softmax exists.")
    with np.errstate(divide="ignore"):
        print("  log(softmax(x)) =", np.log(p)[0])


if __name__ == "__main__":
    print("== the range of exp ==")
    where_exp_dies()
    the_failure()
    underflow_is_the_quiet_one()
