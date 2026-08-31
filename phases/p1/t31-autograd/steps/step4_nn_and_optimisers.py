#!/usr/bin/env python3
"""Step 4 — a layer library and two optimisers, on top of a working engine.

Run:  python3 steps/step4_nn_and_optimisers.py

Once `Tensor` composes, `nn.Linear` is ten lines. What is *not* ten lines of
bookkeeping is initialisation: draw weights from N(0,1) into a stack of tanh
layers and every activation saturates, every gradient vanishes, and the loss
sits flat while you blame the optimiser. This step measures that directly.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t31_autograd import MLP, SGD, Adam, Linear, Sequential, Tanh, Tensor, mse_loss


def initialisation_decides_everything() -> None:
    """Five tanh layers, two scalings — measured, not asserted.

    Careful here, because the folklore is sloppier than the physics. Blowing up
    the weights does *not* automatically shrink the gradient reaching layer 1:
    the backward pass multiplies by W^T too, and the larger W partly cancels
    the smaller tanh'. What it reliably does is **saturate** the units — pin
    them at +/-1, where the layer is a constant function of its input and has
    nothing left to learn with. So the honest measurement is the saturated
    fraction, and then the loss curve the two inits actually produce.
    """
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((256, 16)), requires_grad=False)
    target = Tensor(np.tanh(x.data @ rng.standard_normal((16, 1))), requires_grad=False)

    for label, gain in (("N(0,1) x 5.6  (naive)", 5.6), ("1/sqrt(fan_in)  (Xavier)", 1.0)):
        layers = [Linear(16, 16, nonlinearity="tanh", rng=np.random.default_rng(1)) for _ in range(4)]
        head = Linear(16, 1, nonlinearity="linear", rng=np.random.default_rng(2))
        for lin in layers:
            lin.weight.data *= gain
        net = Sequential(*[m for lin in layers for m in (lin, Tanh())], head)

        h, saturated = x, []
        for layer in net:
            h = layer(h)
            if isinstance(layer, Tanh):
                saturated.append(float((np.abs(h.data) > 0.99).mean()))

        print(f"  {label}")
        print("    saturated units per layer: " + " ".join(f"{s:5.0%}" for s in saturated))
        for opt_name, make_opt in (("SGD(0.1) ", lambda p: SGD(p, lr=0.1)),
                                   ("Adam(0.01)", lambda p: Adam(p, lr=0.01))):
            layers2 = [Linear(16, 16, nonlinearity="tanh", rng=np.random.default_rng(1)) for _ in range(4)]
            head2 = Linear(16, 1, nonlinearity="linear", rng=np.random.default_rng(2))
            for lin in layers2:
                lin.weight.data *= gain
            net2 = Sequential(*[m for lin in layers2 for m in (lin, Tanh())], head2)
            opt = make_opt(net2.parameters())
            first = last = 0.0
            for step in range(300):
                loss = mse_loss(net2(x), target)
                net2.zero_grad()
                loss.backward()
                opt.step()
                if step == 0:
                    first = float(loss.data)
                last = float(loss.data)
            print(f"    {opt_name}  MSE {first:.4f} -> {last:.4f} after 300 steps")


def optimisers_side_by_side() -> None:
    rng = np.random.default_rng(0)
    X = Tensor(rng.standard_normal((128, 6)), requires_grad=False)
    true_w = rng.standard_normal((6, 1))
    y = Tensor(X.data @ true_w + 0.1 * rng.standard_normal((128, 1)), requires_grad=False)

    for name, make in (("SGD(lr=0.05)", lambda p: SGD(p, lr=0.05)),
                       ("SGD+momentum", lambda p: SGD(p, lr=0.05, momentum=0.9)),
                       ("Adam(lr=0.05)", lambda p: Adam(p, lr=0.05))):
        model = MLP([6, 16, 1], rng=np.random.default_rng(0))
        opt = make(model.parameters())
        for _ in range(150):
            loss = mse_loss(model(X), y)
            model.zero_grad()
            loss.backward()
            opt.step()
        print(f"  {name:<16} final MSE {float(loss.data):.5f}")


if __name__ == "__main__":
    print("== why initialisation is not a detail ==")
    initialisation_decides_everything()
    print("\n== three optimisers, same graph ==")
    optimisers_side_by_side()
