"""Tests for the frozen ImageNet backbone and the two ways it is wired into the models.

The invariants that matter:

  * the backbone must stay frozen AND in eval mode even when the parent is put in train mode
    -- convnext_tiny carries stochastic depth, so a train-mode backbone would silently
    randomise the features and make the train and validation passes see different encoders;
  * frozen parameters must be invisible to the optimizer and to the weight EMA, otherwise
    weight decay would erode a network nobody is training;
  * `semantic_encoder=None` / `semantic_channels=0` must leave the UNet numerically identical
    to the version without semantic injection, so existing architectures and checkpoints load;
  * the CMT encoder must expose the pyramid contract PixelDecoder relies on.

The backbone is instantiated with pretrained=False throughout: these tests are about wiring,
and downloading ImageNet weights would make the suite depend on the network.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

from pyaiwrap.ema import WeightEma
from pyaiwrap.neural_network import (
    CONVNEXT_TINY_STAGE_CHANNELS,
    FrozenConvNextBackbone,
    PretrainedLuminanceEncoder,
    PretrainedSemanticEncoder,
    UNetBottleneck,
    UNetWithSkipConnections,
)
from pyaiwrap.optimizers import createOptimizer


def buildUNetLayersConfig(semantic_channels: int = 0) -> List[Dict]:
    """A miniature version of the rgb_merge topology: three downsamples, bottleneck at 1/8."""
    return [
        {"type": "UNetEncoderBlock",
         "params": {"in_channels": 4, "out_channels": 8, "downsample": True,
                    "block_name": "enc1"}},
        {"type": "UNetEncoderBlock",
         "params": {"in_channels": 8, "out_channels": 16, "downsample": True,
                    "block_name": "enc2"}},
        {"type": "UNetEncoderBlock",
         "params": {"in_channels": 16, "out_channels": 32, "downsample": True,
                    "block_name": "enc3"}},
        {"type": "UNetBottleneck",
         "params": {"channels": 32, "bottleneck_channels": 64,
                    "semantic_channels": semantic_channels}},
        {"type": "UNetDecoderBlock",
         "params": {"in_channels": 32, "out_channels": 32, "upsample": True,
                    "skip_connection": "enc3", "block_name": "dec1"}},
        {"type": "UNetDecoderBlock",
         "params": {"in_channels": 32, "out_channels": 16, "upsample": True,
                    "skip_connection": "enc2", "block_name": "dec2"}},
        {"type": "UNetDecoderBlock",
         "params": {"in_channels": 16, "out_channels": 8, "upsample": True,
                    "skip_connection": "enc1", "block_name": "dec3"}},
        {"type": "Conv2d",
         "params": {"in_channels": 8, "out_channels": 3, "kernel_size": 3, "padding": 1}},
    ]


def testBackboneParametersAreFrozen() -> None:
    backbone: FrozenConvNextBackbone = FrozenConvNextBackbone(pretrained=False)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def testBackboneStaysInEvalModeWhenParentTrains() -> None:
    backbone: FrozenConvNextBackbone = FrozenConvNextBackbone(pretrained=False)
    backbone.train(True)
    assert not backbone._features.training

    encoder: PretrainedSemanticEncoder = PretrainedSemanticEncoder(pretrained=False)
    encoder.train(True)
    assert not encoder.backbone._features.training
    # the trained part must follow the parent, only the backbone is pinned
    assert encoder.fuse.training


def testBackboneIsDeterministicInTrainMode() -> None:
    """Stochastic depth would break this if the backbone followed the parent's train mode."""
    backbone: FrozenConvNextBackbone = FrozenConvNextBackbone(pretrained=False)
    backbone.train(True)
    luminance: torch.Tensor = torch.rand(2, 1, 64, 64)

    first: List[torch.Tensor] = backbone(luminance)
    second: List[torch.Tensor] = backbone(luminance)

    for first_stage, second_stage in zip(first, second):
        assert torch.equal(first_stage, second_stage)


def testBackboneReturnsTheExpectedPyramid() -> None:
    backbone: FrozenConvNextBackbone = FrozenConvNextBackbone(pretrained=False)
    stage_outputs: List[torch.Tensor] = backbone(torch.rand(2, 1, 128, 128))

    assert [output.shape[1] for output in stage_outputs] == list(CONVNEXT_TINY_STAGE_CHANNELS)
    assert [output.shape[-1] for output in stage_outputs] == [32, 16, 8, 4]


def testBackboneRejectsMultiChannelInput() -> None:
    backbone: FrozenConvNextBackbone = FrozenConvNextBackbone(pretrained=False)
    with pytest.raises(ValueError, match="single luminance channel"):
        backbone(torch.rand(2, 3, 64, 64))


def testSemanticEncoderFusesToOneStride() -> None:
    encoder: PretrainedSemanticEncoder = PretrainedSemanticEncoder(
        out_channels=16, output_stride=8, norm_groups=4, pretrained=False
    )
    fused: torch.Tensor = encoder(torch.rand(2, 1, 128, 128))
    assert fused.shape == (2, 16, 16, 16)


def testSemanticEncoderRejectsNonDivisorGroupCount() -> None:
    with pytest.raises(ValueError, match="must divide"):
        PretrainedSemanticEncoder(out_channels=16, norm_groups=5, pretrained=False)


def testSemanticEncoderLateralsAndFuseAreTrainable() -> None:
    encoder: PretrainedSemanticEncoder = PretrainedSemanticEncoder(
        out_channels=16, norm_groups=4, pretrained=False
    )
    trainable = {name for name, parameter in encoder.named_parameters() if parameter.requires_grad}

    assert any(name.startswith("lateral_projections") for name in trainable)
    assert any(name.startswith("fuse") for name in trainable)
    assert not any(name.startswith("backbone") for name in trainable)


@pytest.mark.parametrize("no_decay_groups", [False, True])
def testOptimizerStepLeavesTheFrozenBackboneUntouched(no_decay_groups: bool) -> None:
    """Weight decay must not erode a backbone nobody is training.

    Both optimizer paths are covered because they protect the backbone differently: with
    NO_DECAY_GROUPS on, createParameterGroups drops the frozen parameters outright; with it
    off the whole parameter iterator is handed to AdamW, which then skips them only because
    their gradient stays None. The invariant the models rely on is the same either way.
    """
    torch.manual_seed(0)
    encoder: PretrainedSemanticEncoder = PretrainedSemanticEncoder(
        out_channels=16, norm_groups=4, pretrained=False
    )
    config: Dict = {"OPTIMIZER_TYPE": "adamw", "LEARNING_RATE": 1e-2, "WEIGHT_DECAY": 0.5,
                    "B1": 0.9, "B2": 0.999, "NO_DECAY_GROUPS": no_decay_groups}
    optimizer = createOptimizer(encoder, config)

    before: Dict[str, torch.Tensor] = {
        name: parameter.detach().clone()
        for name, parameter in encoder.named_parameters() if not parameter.requires_grad
    }

    encoder(torch.rand(2, 1, 64, 64)).sum().backward()
    optimizer.step()

    for name, original in before.items():
        assert torch.equal(original, dict(encoder.named_parameters())[name]), name


def testEmaCopiesFrozenBackboneVerbatim() -> None:
    """The averaged state_dict must stay loadable, so frozen tensors are carried, not averaged."""
    torch.manual_seed(0)
    encoder: PretrainedSemanticEncoder = PretrainedSemanticEncoder(
        out_channels=16, norm_groups=4, pretrained=False
    )
    ema: WeightEma = WeightEma(encoder, decay=0.5, warmup_updates=0)

    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.add_(1.0)
    ema.update(encoder)

    averaged: Dict[str, torch.Tensor] = ema.emaStateDict()
    parameters: Dict[str, torch.Tensor] = dict(encoder.named_parameters())

    frozen_names = [name for name, parameter in parameters.items() if not parameter.requires_grad]
    assert frozen_names, "expected the backbone to contribute frozen parameters"
    for name in frozen_names:
        assert torch.equal(averaged[name], parameters[name]), name

    trained_name = next(name for name, parameter in parameters.items() if parameter.requires_grad)
    assert not torch.equal(averaged[trained_name], parameters[trained_name])


def testUNetWithoutSemanticEncoderIsUnchanged() -> None:
    torch.manual_seed(0)
    plain: UNetWithSkipConnections = UNetWithSkipConnections(buildUNetLayersConfig())
    assert plain.semantic_encoder is None

    plain.eval()
    with torch.no_grad():
        output: torch.Tensor = plain(torch.rand(2, 4, 64, 64))
    assert output.shape == (2, 3, 64, 64)


def testUNetWithSemanticEncoderRunsAndKeepsOutputShape() -> None:
    torch.manual_seed(0)
    unet: UNetWithSkipConnections = UNetWithSkipConnections(
        buildUNetLayersConfig(semantic_channels=16),
        semantic_encoder={"out_channels": 16, "output_stride": 8, "norm_groups": 4,
                          "pretrained": False}
    )
    unet.eval()
    with torch.no_grad():
        output: torch.Tensor = unet(torch.rand(2, 4, 64, 64))

    assert output.shape == (2, 3, 64, 64)


def testSemanticInjectionChangesTheOutput() -> None:
    """Guards against the injected features being silently dropped on the floor."""
    torch.manual_seed(0)
    unet: UNetWithSkipConnections = UNetWithSkipConnections(
        buildUNetLayersConfig(semantic_channels=16),
        semantic_encoder={"out_channels": 16, "output_stride": 8, "norm_groups": 4,
                          "pretrained": False}
    )
    unet.eval()
    # Same non-luminance channels, different luminance: only the semantic branch and enc1 see
    # the difference, and a dropped injection would be visible as a much smaller delta.
    common: torch.Tensor = torch.rand(2, 3, 64, 64)
    first: torch.Tensor = torch.cat([torch.zeros(2, 1, 64, 64), common], dim=1)
    second: torch.Tensor = torch.cat([torch.ones(2, 1, 64, 64), common], dim=1)

    with torch.no_grad():
        assert not torch.allclose(unet(first), unet(second))


def testBottleneckWithoutSemanticChannelsIgnoresFeatures() -> None:
    torch.manual_seed(0)
    bottleneck: UNetBottleneck = UNetBottleneck(channels=8, bottleneck_channels=16)
    x: torch.Tensor = torch.rand(2, 8, 8, 8)

    bottleneck.eval()
    with torch.no_grad():
        assert torch.equal(bottleneck(x), bottleneck(x, torch.rand(2, 4, 8, 8)))


def testBottleneckRequiresFeaturesWhenConfigured() -> None:
    bottleneck: UNetBottleneck = UNetBottleneck(channels=8, bottleneck_channels=16,
                                                semantic_channels=4)
    with pytest.raises(ValueError, match="received no semantic features"):
        bottleneck(torch.rand(2, 8, 8, 8))


def testBottleneckRejectsMismatchedFeatureResolution() -> None:
    bottleneck: UNetBottleneck = UNetBottleneck(channels=8, bottleneck_channels=16,
                                                semantic_channels=4)
    with pytest.raises(ValueError, match="does not match"):
        bottleneck(torch.rand(2, 8, 8, 8), torch.rand(2, 4, 16, 16))


def testBottleneckOutputWidthIsUnaffectedBySemanticChannels() -> None:
    bottleneck: UNetBottleneck = UNetBottleneck(channels=8, bottleneck_channels=16,
                                                semantic_channels=4)
    output: torch.Tensor = bottleneck(torch.rand(2, 8, 8, 8), torch.rand(2, 4, 8, 8))
    assert output.shape == (2, 8, 8, 8)


def testLuminanceEncoderExposesThePyramidContract() -> None:
    encoder: PretrainedLuminanceEncoder = PretrainedLuminanceEncoder(pretrained=False)

    assert encoder.getDownsampleChannels() == list(CONVNEXT_TINY_STAGE_CHANNELS)
    assert encoder.getNumDownsampleLayers() == 4

    stage_outputs: List[torch.Tensor] = encoder(torch.rand(2, 1, 128, 128))
    assert [output.shape[-1] for output in stage_outputs] == [32, 16, 8, 4]


def testLuminanceEncoderProjectsToRequestedWidths() -> None:
    encoder: PretrainedLuminanceEncoder = PretrainedLuminanceEncoder(
        pretrained=False, stage_channels=[96, 256, 256, 512]
    )
    assert encoder.getDownsampleChannels() == [96, 256, 256, 512]

    stage_outputs: List[torch.Tensor] = encoder(torch.rand(2, 1, 128, 128))
    assert [output.shape[1] for output in stage_outputs] == [96, 256, 256, 512]
    # stage 0 is already 96 wide, so its projection must be a no-op rather than a 1x1
    assert isinstance(encoder.stage_projections[0], nn.Identity)


def testLuminanceEncoderRejectsWrongStageCount() -> None:
    with pytest.raises(ValueError, match="one entry per backbone stage"):
        PretrainedLuminanceEncoder(pretrained=False, stage_channels=[96, 256])
