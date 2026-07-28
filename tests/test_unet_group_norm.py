"""Tests for the optional GroupNorm in the UNet blocks.

The critical invariant is backward compatibility: `norm_groups=None` (the default) must
leave every block numerically identical to the pre-normalisation version, so existing
architectures and checkpoints are unaffected. Beyond that, an enabled norm must actually
normalise and must reject a group count that does not divide the channels.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Dict

import pytest
import torch
import torch.nn as nn

from pyaiwrap.neural_network import (
    UNetBottleneck,
    UNetDecoderBlock,
    UNetEncoderBlock,
    createGroupNorm,
)


def testCreateGroupNormDisabledIsIdentity() -> None:
    norm: nn.Module = createGroupNorm(None, 64)
    assert isinstance(norm, nn.Identity)


def testCreateGroupNormRejectsNonDivisorGroupCount() -> None:
    with pytest.raises(ValueError, match="must divide"):
        createGroupNorm(32, 48)


def testCreateGroupNormNormalisesActivations() -> None:
    norm: nn.Module = createGroupNorm(2, 4)
    x: torch.Tensor = torch.randn(3, 4, 5, 5) * 7.0 + 3.0
    out: torch.Tensor = norm(x)

    grouped: torch.Tensor = out.reshape(3, 2, -1)
    assert torch.allclose(grouped.mean(dim=-1), torch.zeros(3, 2), atol=1e-4)
    assert torch.allclose(grouped.std(dim=-1, unbiased=False), torch.ones(3, 2), atol=1e-3)


def testEncoderBlockWithoutNormIsUnchanged() -> None:
    """No norm layers, no extra parameters, and the norms are true no-ops."""
    torch.manual_seed(0)
    block: UNetEncoderBlock = UNetEncoderBlock(in_channels=3, out_channels=8, downsample=True)
    assert isinstance(block.norm1, nn.Identity)
    assert isinstance(block.norm2, nn.Identity)

    x: torch.Tensor = torch.randn(2, 3, 16, 16)
    block.eval()
    with torch.no_grad():
        out: torch.Tensor = block(x)
    assert out.shape == (2, 8, 8, 8)


def testNormAddsOnlyNormParameters() -> None:
    """Enabling the norm must not disturb the conv weights, only add affine parameters."""
    torch.manual_seed(0)
    plain: UNetEncoderBlock = UNetEncoderBlock(in_channels=3, out_channels=8, downsample=False)
    torch.manual_seed(0)
    normed: UNetEncoderBlock = UNetEncoderBlock(in_channels=3, out_channels=8, downsample=False,
                                                norm_groups=4)

    plain_params: Dict[str, torch.Tensor] = dict(plain.named_parameters())
    normed_params: Dict[str, torch.Tensor] = dict(normed.named_parameters())
    for name, param in plain_params.items():
        assert torch.equal(param, normed_params[name]), name

    added: set = set(normed_params) - set(plain_params)
    assert added == {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"}


@pytest.mark.parametrize("norm_groups", [None, 4])
def testDecoderBlockRunsWithAndWithoutNorm(norm_groups) -> None:
    block: UNetDecoderBlock = UNetDecoderBlock(in_channels=8, out_channels=8, upsample=False,
                                               skip_connection="enc1", block_name="dec1",
                                               norm_groups=norm_groups)
    x: torch.Tensor = torch.randn(2, 8, 16, 16)
    encoder_features: Dict[str, torch.Tensor] = {"enc1": torch.randn(2, 8, 16, 16)}
    out: torch.Tensor = block(x, encoder_features)
    assert out.shape == (2, 8, 16, 16)


@pytest.mark.parametrize("norm_groups", [None, 4])
def testBottleneckRunsWithAndWithoutNorm(norm_groups) -> None:
    bottleneck: UNetBottleneck = UNetBottleneck(channels=8, bottleneck_channels=16,
                                                norm_groups=norm_groups)
    out: torch.Tensor = bottleneck(torch.randn(2, 8, 8, 8))
    assert out.shape == (2, 8, 8, 8)


def testBottleneckRejectsNonDivisorGroupCount() -> None:
    with pytest.raises(ValueError, match="must divide"):
        UNetBottleneck(channels=8, bottleneck_channels=12, norm_groups=8)
