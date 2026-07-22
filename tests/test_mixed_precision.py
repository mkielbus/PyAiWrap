"""Tests for the bf16 autocast path in GeneratorColorizationLoss.

The generator forward may run under torch.autocast, but every loss term stays fp32:
labToRgb and the colourfulness statistic are precision-sensitive and LPIPS is a frozen
fp32 net. The default (MIXED_PRECISION off) must leave the fp32 path completely untouched,
which is why the implementation uses contextlib.nullcontext rather than
torch.autocast(enabled=False) -- merely entering a disabled autocast region perturbs
kernel selection by about one fp32 ULP.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn

from pyaiwrap.loss import GeneratorColorizationLoss

IMAGE_SIZE: int = 32
BATCH_SIZE: int = 2


class _RecordingGenerator(nn.Module):
    """Conv net that records the dtype its convolution actually ran in."""

    def __init__(self, out_channels: int = 2) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(1, out_channels, 3, padding=1)
        self.seen_dtype: torch.dtype = torch.float32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.tanh(self.conv(x))
        self.seen_dtype = out.dtype
        return out


class _FakeMetrics:
    def __init__(self) -> None:
        self.records: List[Dict[str, float]] = []

    def accumulate(self, values: Dict[str, float]) -> None:
        self.records.append(values)


def _buildLoss(mixed_precision: bool, dtype: str = "bfloat16") -> GeneratorColorizationLoss:
    return GeneratorColorizationLoss(
        reconstruction_loss_fn=nn.L1Loss(), recon_weight=1.0,
        perceptual_weight=0.0, colorfulness_weight=0.015, colorfulness_target=None,
        use_lpips=False, device=torch.device("cpu"),
        input_channel="luminance", target_channel="AB",
        mixed_precision=mixed_precision, mixed_precision_dtype=dtype)


def _batch() -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return (torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE),
            torch.rand(BATCH_SIZE, 2, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1)


def testDefaultIsOff() -> None:
    """A loss built without the flag must not autocast anything."""
    loss_fn = GeneratorColorizationLoss(device=torch.device("cpu"))
    assert loss_fn.mixed_precision is False


def testReconstructedOutputStaysFloat32WhenMixed() -> None:
    """The generator may emit bf16, but what the loss terms consume must be fp32."""
    torch.manual_seed(0)
    generator = _RecordingGenerator()
    loss_fn = _buildLoss(mixed_precision=True)
    output = loss_fn({"generator": generator}, _batch(), _FakeMetrics(), None)
    assert output["reconstructed_images"].dtype == torch.float32


def testAutocastActuallyEngagesTheGenerator() -> None:
    """Guard against the flag silently doing nothing on the forward pass."""
    torch.manual_seed(0)
    generator = _RecordingGenerator()
    _buildLoss(mixed_precision=True)({"generator": generator}, _batch(), _FakeMetrics(), None)
    assert generator.seen_dtype == torch.bfloat16

    torch.manual_seed(0)
    plain = _RecordingGenerator()
    _buildLoss(mixed_precision=False)({"generator": plain}, _batch(), _FakeMetrics(), None)
    assert plain.seen_dtype == torch.float32


def testMixedPrecisionLossStaysCloseToFp32() -> None:
    """bf16 forward must not move the loss more than a fraction of a percent."""
    values: Dict[bool, float] = {}
    for mixed in (False, True):
        torch.manual_seed(0)
        generator = _RecordingGenerator()
        output = _buildLoss(mixed_precision=mixed)(
            {"generator": generator}, _batch(), _FakeMetrics(), None)
        values[mixed] = float(output["loss"].detach())
        assert torch.isfinite(output["loss"]).item()
    assert abs(values[True] - values[False]) / abs(values[False]) < 0.01


def testGradientsFlowThroughTheAutocastRegion() -> None:
    torch.manual_seed(0)
    generator = _RecordingGenerator()
    loss_fn = _buildLoss(mixed_precision=True)
    loss_fn({"generator": generator}, _batch(), _FakeMetrics(), None)
    assert generator.conv.weight.grad is not None
    assert torch.isfinite(generator.conv.weight.grad).all()
    assert generator.conv.weight.grad.dtype == torch.float32


def testFloat16IsAccepted() -> None:
    torch.manual_seed(0)
    generator = _RecordingGenerator()
    _buildLoss(mixed_precision=True, dtype="float16")(
        {"generator": generator}, _batch(), _FakeMetrics(), None)
    assert generator.seen_dtype == torch.float16


def testUnknownDtypeIsRejected() -> None:
    with pytest.raises(ValueError, match="mixed_precision_dtype"):
        _buildLoss(mixed_precision=True, dtype="float8")
