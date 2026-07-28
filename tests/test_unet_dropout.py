"""Tests for block-level dropout in the UNet.

The bug this guards against: `UNetWithSkipConnections` buckets layers by type and applies
everything that is not an encoder/decoder/bottleneck *after* the whole decoder, so a
standalone `Dropout2d` entry in the architecture JSON silently moves to the output no
matter where it was written. The `dropout` parameter on the bottleneck and decoder blocks
puts it at a defined place inside the block instead.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from pyaiwrap.neural_network import (
    UNetBottleneck,
    UNetDecoderBlock,
    UNetWithSkipConnections,
    createDropout2d,
)


def testCreateDropoutDisabledIsIdentity() -> None:
    assert isinstance(createDropout2d(None), nn.Identity)
    assert isinstance(createDropout2d(0.0), nn.Identity)


def testCreateDropoutRejectsOutOfRangeProbability() -> None:
    with pytest.raises(ValueError, match="must be in"):
        createDropout2d(1.0)
    with pytest.raises(ValueError, match="must be in"):
        createDropout2d(-0.1)


def testBottleneckDropoutZeroesWholeChannelsInTrainMode() -> None:
    torch.manual_seed(0)
    bottleneck: UNetBottleneck = UNetBottleneck(channels=64, bottleneck_channels=64, dropout=0.5)
    bottleneck.train()
    out: torch.Tensor = bottleneck(torch.randn(4, 64, 8, 8))

    # Dropout2d drops entire feature maps, so a dropped channel is all-zero.
    channel_sums: torch.Tensor = out.abs().sum(dim=(2, 3))
    assert (channel_sums == 0).any(), "expected at least one dropped channel"
    assert (channel_sums == 0).all(dim=1).logical_not().all(), "expected some channels kept"


def testBottleneckDropoutIsDisabledInEvalMode() -> None:
    torch.manual_seed(0)
    bottleneck: UNetBottleneck = UNetBottleneck(channels=16, bottleneck_channels=32, dropout=0.5)
    bottleneck.eval()
    x: torch.Tensor = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        first: torch.Tensor = bottleneck(x)
        second: torch.Tensor = bottleneck(x)

    assert torch.equal(first, second)
    assert (first.abs().sum(dim=(2, 3)) > 0).all()


def testDecoderBlockDropoutIsDisabledInEvalMode() -> None:
    torch.manual_seed(0)
    block: UNetDecoderBlock = UNetDecoderBlock(in_channels=16, out_channels=16, upsample=False,
                                               skip_connection="enc1", block_name="dec0",
                                               dropout=0.5)
    block.eval()
    x: torch.Tensor = torch.randn(2, 16, 8, 8)
    features: Dict[str, torch.Tensor] = {"enc1": torch.randn(2, 16, 8, 8)}
    with torch.no_grad():
        assert torch.equal(block(x, features), block(x, features))


def testBlocksDefaultToNoDropout() -> None:
    assert isinstance(UNetBottleneck(channels=8, bottleneck_channels=8).dropout, nn.Identity)
    assert isinstance(
        UNetDecoderBlock(in_channels=8, out_channels=8, upsample=False).dropout, nn.Identity
    )


def _buildUNet(layers_config: List[Dict[str, Any]]) -> UNetWithSkipConnections:
    return UNetWithSkipConnections(layers_config=layers_config)


STANDALONE_DROPOUT_CONFIG: List[Dict[str, Any]] = [
    {"type": "UNetEncoderBlock",
     "params": {"in_channels": 3, "out_channels": 8, "downsample": False, "block_name": "enc1"}},
    {"type": "UNetBottleneck", "params": {"channels": 8, "bottleneck_channels": 8}},
    {"type": "Dropout2d", "params": {"p": 0.3}},
    {"type": "UNetDecoderBlock",
     "params": {"in_channels": 8, "out_channels": 8, "upsample": False,
                "skip_connection": "enc1", "block_name": "dec0"}},
    {"type": "Conv2d", "params": {"in_channels": 8, "out_channels": 3, "kernel_size": 1}},
]


def testStandaloneDropoutInJsonDoesNotStayInPosition() -> None:
    """Documents the trap: a Dropout2d written mid-architecture is applied at the end.

    It is written between the bottleneck and the decoder, but it lands in `other_layers`
    alongside the output conv, i.e. after the whole decoder. This is why the block-level
    `dropout` parameter exists; do not put Dropout2d in a UNet architecture JSON.
    """
    unet: UNetWithSkipConnections = _buildUNet(STANDALONE_DROPOUT_CONFIG)

    assert isinstance(unet.bottleneck.dropout, nn.Identity)
    assert isinstance(unet.decoder_blocks["dec0"].dropout, nn.Identity)

    trailing: List[str] = [type(layer._layer).__name__ for layer in unet.other_layers]
    assert trailing == ["Dropout2d", "Conv2d"], trailing


def testBlockLevelDropoutStaysInsideTheBlock() -> None:
    config: List[Dict[str, Any]] = [
        {"type": "UNetEncoderBlock",
         "params": {"in_channels": 3, "out_channels": 8, "downsample": False, "block_name": "enc1"}},
        {"type": "UNetBottleneck",
         "params": {"channels": 8, "bottleneck_channels": 8, "dropout": 0.3}},
        {"type": "UNetDecoderBlock",
         "params": {"in_channels": 8, "out_channels": 8, "upsample": False,
                    "skip_connection": "enc1", "block_name": "dec0", "dropout": 0.15}},
        {"type": "Conv2d", "params": {"in_channels": 8, "out_channels": 3, "kernel_size": 1}},
    ]
    unet: UNetWithSkipConnections = _buildUNet(config)

    assert isinstance(unet.bottleneck.dropout, nn.Dropout2d)
    assert unet.bottleneck.dropout.p == pytest.approx(0.3)
    assert unet.decoder_blocks["dec0"].dropout.p == pytest.approx(0.15)
    # Nothing but the output conv trails the decoder.
    assert [type(layer._layer).__name__ for layer in unet.other_layers] == ["Conv2d"]
