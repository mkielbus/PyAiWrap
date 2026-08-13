"""Tests that the loss reads lightness from a conditioned model input, not from the fallback.

A chroma-output model needs the input's luminance to render its prediction as RGB for the
perceptual and colorfulness terms. The width tests that pick that path used to demand exactly
one channel, so a segmentation-conditioned input ([luminance, boundary, area]) missed every
branch and landed on the middle-gray fallback. Nothing raises: the run simply optimises a
perceptual loss computed against flat L* = 50, which is a wrong number that looks like a
plausible one.

Naming/style follows the project convention (see CLAUDE.md).
"""
import pytest
import torch

from pyaiwrap.loss import GeneratorColorizationLoss

BATCH: int = 2
SIZE: int = 8


def _buildLoss(target_channel: str) -> GeneratorColorizationLoss:
    return GeneratorColorizationLoss(
        reconstruction_loss_fn=torch.nn.L1Loss(),
        target_channel=target_channel,
        input_channel="luminance",
        device=torch.device("cpu")
    )


@pytest.mark.parametrize("target_channel", ["AB", "LAB_A", "LAB_B"])
def testConditionedInputUsesItsLuminance(target_channel: str) -> None:
    """The conditioned input must render identically to the bare luminance it starts with."""
    loss_fn: GeneratorColorizationLoss = _buildLoss(target_channel)
    channels: int = 2 if target_channel == "AB" else 1
    prediction: torch.Tensor = torch.rand(BATCH, channels, SIZE, SIZE) * 60.0 - 30.0

    luminance: torch.Tensor = torch.rand(BATCH, 1, SIZE, SIZE)
    conditioned: torch.Tensor = torch.cat([luminance, torch.rand(BATCH, 2, SIZE, SIZE)], dim=1)

    assert torch.allclose(loss_fn._convertToRgbForLoss(prediction, conditioned),
                          loss_fn._convertToRgbForLoss(prediction, luminance))


@pytest.mark.parametrize("target_channel", ["AB", "LAB_A", "LAB_B"])
def testConditionedInputDoesNotHitTheGrayFallback(target_channel: str) -> None:
    """The fallback is L* = 50 everywhere, so a varying luminance must give a different image."""
    loss_fn: GeneratorColorizationLoss = _buildLoss(target_channel)
    channels: int = 2 if target_channel == "AB" else 1
    prediction: torch.Tensor = torch.zeros(BATCH, channels, SIZE, SIZE)

    dark: torch.Tensor = torch.cat([torch.full((BATCH, 1, SIZE, SIZE), 0.1),
                                    torch.rand(BATCH, 2, SIZE, SIZE)], dim=1)
    bright: torch.Tensor = torch.cat([torch.full((BATCH, 1, SIZE, SIZE), 0.9),
                                      torch.rand(BATCH, 2, SIZE, SIZE)], dim=1)

    assert not torch.allclose(loss_fn._convertToRgbForLoss(prediction, dark),
                              loss_fn._convertToRgbForLoss(prediction, bright))


def testUnconditionedInputIsUnchanged() -> None:
    """The pre-existing single-channel path must render exactly as it did before."""
    loss_fn: GeneratorColorizationLoss = _buildLoss("AB")
    prediction: torch.Tensor = torch.rand(BATCH, 2, SIZE, SIZE) * 60.0 - 30.0
    luminance: torch.Tensor = torch.rand(BATCH, 1, SIZE, SIZE)

    from pyaiwrap.transforms import labToRgb, luminanceToLabRange
    expected: torch.Tensor = labToRgb(luminanceToLabRange(luminance, "srgb"), prediction)

    assert torch.allclose(loss_fn._convertToRgbForLoss(prediction, luminance), expected)


def testRgbTargetIgnoresTheInputEntirely() -> None:
    """The RGB path returns the prediction untouched, conditioned input or not (this is v9)."""
    loss_fn: GeneratorColorizationLoss = _buildLoss("RGB")
    prediction: torch.Tensor = torch.rand(BATCH, 3, SIZE, SIZE)

    assert torch.equal(
        loss_fn._convertToRgbForLoss(prediction, torch.rand(BATCH, 3, SIZE, SIZE)), prediction)
