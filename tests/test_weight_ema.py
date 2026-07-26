"""Tests for WeightEma.

The invariant that matters most: EMA never perturbs training. applyTo/restore must return
the model's raw weights bit-for-bit, floating-point parameters and BatchNorm running stats
are averaged, and integer buffers (num_batches_tracked) are copied verbatim so the shadow
stays a complete, loadable state_dict.

Naming/style follows the project convention (see CLAUDE.md).
"""
import copy

import torch
import torch.nn as nn

from pyaiwrap.ema import WeightEma


class _TinyNet(nn.Module):
    """Conv + BatchNorm so both parameters and running-stat buffers are exercised."""

    def __init__(self) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(1, 2, 3, padding=1)
        self.norm: nn.BatchNorm2d = nn.BatchNorm2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


def _setAll(model: nn.Module, value: float) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)


def testUpdateAveragesParameters() -> None:
    model: _TinyNet = _TinyNet()
    _setAll(model, 1.0)
    ema: WeightEma = WeightEma(model, decay=0.9, warmup_updates=0)

    _setAll(model, 2.0)
    ema.update(model)

    # shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1 for every trainable parameter.
    averaged = ema.emaStateDict()
    assert torch.allclose(averaged["conv.weight"], torch.full_like(averaged["conv.weight"], 1.1))
    assert torch.allclose(averaged["conv.bias"], torch.full_like(averaged["conv.bias"], 1.1))


def testDecayWarmupRamp() -> None:
    model: _TinyNet = _TinyNet()
    _setAll(model, 1.0)
    ema: WeightEma = WeightEma(model, decay=0.9, warmup_updates=2)

    _setAll(model, 2.0)
    ema.update(model)  # num_updates=1 -> decay = 0.9 * 1/2 = 0.45

    # shadow = 0.45 * 1.0 + 0.55 * 2.0 = 1.55
    averaged = ema.emaStateDict()
    assert torch.allclose(averaged["conv.bias"], torch.full_like(averaged["conv.bias"], 1.55))


def testApplyRestoreIsBitExact() -> None:
    model: _TinyNet = _TinyNet()
    ema: WeightEma = WeightEma(model, decay=0.9)

    # Drive the shadow away from the live weights.
    _setAll(model, 5.0)
    for _ in range(3):
        ema.update(model)

    raw_state = copy.deepcopy(model.state_dict())
    ema.applyTo(model)
    # While applied, the model holds the (different) averaged weights.
    assert not torch.allclose(model.conv.weight, raw_state["conv.weight"])

    ema.restore(model)
    restored_state = model.state_dict()
    for name, tensor in raw_state.items():
        assert torch.equal(restored_state[name], tensor), f"{name} not restored exactly"


def testIntegerBuffersCopiedNotAveraged() -> None:
    model: _TinyNet = _TinyNet()
    ema: WeightEma = WeightEma(model, decay=0.9)

    # num_batches_tracked is an int buffer BatchNorm bumps during a forward in train mode.
    model.train()
    model(torch.randn(4, 1, 8, 8))
    ema.update(model)

    averaged = ema.emaStateDict()
    assert averaged["norm.num_batches_tracked"].dtype == model.norm.num_batches_tracked.dtype
    assert torch.equal(averaged["norm.num_batches_tracked"], model.norm.num_batches_tracked)


def testStateDictRoundTrip() -> None:
    model: _TinyNet = _TinyNet()
    _setAll(model, 1.0)
    ema: WeightEma = WeightEma(model, decay=0.9)
    _setAll(model, 3.0)
    for _ in range(5):
        ema.update(model)

    saved = ema.stateDict()
    reloaded: WeightEma = WeightEma(model, decay=0.5)
    reloaded.loadStateDict(saved)

    a = ema.emaStateDict()
    b = reloaded.emaStateDict()
    for name in a:
        assert torch.equal(a[name], b[name])


def testEmaStateDictLoadsIntoModel() -> None:
    model: _TinyNet = _TinyNet()
    ema: WeightEma = WeightEma(model, decay=0.9)
    _setAll(model, 4.0)
    ema.update(model)

    # A fresh model must accept the shadow as a plain state_dict.
    fresh: _TinyNet = _TinyNet()
    fresh.load_state_dict(ema.emaStateDict())
