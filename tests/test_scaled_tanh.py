"""Tests for ScaledTanh, the bounded AB output activation.

It must equal scale*tanh(x), never exceed +/-scale (so the regressed AB channels stay inside
Kornia's Lab gamut) and preserve tensor shape.

Naming/style follows the project convention (see CLAUDE.md).
"""
import torch

from pyaiwrap.neural_network import ScaledTanh


def testMatchesScaledTanhFormula() -> None:
    layer: ScaledTanh = ScaledTanh(scale=127.0)
    x: torch.Tensor = torch.linspace(-50.0, 50.0, 401)
    assert torch.allclose(layer(x), 127.0 * torch.tanh(x))


def testOutputIsBounded() -> None:
    layer: ScaledTanh = ScaledTanh(scale=127.0)
    x: torch.Tensor = torch.linspace(-1000.0, 1000.0, 501)
    assert torch.all(layer(x).abs() <= 127.0)


def testShapePreserved() -> None:
    layer: ScaledTanh = ScaledTanh()
    x: torch.Tensor = torch.randn(3, 2, 16, 16)
    assert layer(x).shape == x.shape


def testDefaultScaleMatchesAbGamut() -> None:
    assert ScaledTanh().scale == 127.0
