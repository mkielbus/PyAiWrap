"""The multi-level semantic pyramid and its routing into the UNet decoder."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from pyaiwrap.neural_network import (FrozenConvNextBackbone, MultiScaleSemanticEncoder,
                                     UNetWithSkipConnections)


def buildLayers(semantic_by_block: Dict[str, int]) -> List[Dict[str, Any]]:
    """The shape rgb_merge uses: three downsampling stages, a stride-8 bottleneck, four
    decoder blocks running at strides 8, 4, 2 and 1."""
    def encoder(name: str, in_channels: int, out_channels: int,
                downsample: bool) -> Dict[str, Any]:
        return {"type": "UNetEncoderBlock", "params": {
            "in_channels": in_channels, "out_channels": out_channels,
            "downsample": downsample, "block_name": name, "norm_groups": 32}}

    def decoder(name: str, in_channels: int, out_channels: int, upsample: bool,
                skip: str) -> Dict[str, Any]:
        return {"type": "UNetDecoderBlock", "params": {
            "in_channels": in_channels, "out_channels": out_channels, "upsample": upsample,
            "skip_connection": skip, "block_name": name, "norm_groups": 32,
            "semantic_channels": semantic_by_block.get(name, 0)}}

    return [
        encoder("enc1", 1, 64, True),
        encoder("enc2", 64, 128, True),
        encoder("enc3", 128, 256, True),
        encoder("enc4", 256, 512, False),
        {"type": "UNetBottleneck", "params": {"channels": 512, "bottleneck_channels": 512,
                                              "norm_groups": 32}},
        decoder("dec0", 512, 512, False, "enc4"),
        decoder("dec1", 512, 256, True, "enc3"),
        decoder("dec2", 256, 128, True, "enc2"),
        decoder("dec3", 128, 64, True, "enc1"),
        {"type": "Conv2d", "params": {"in_channels": 64, "out_channels": 2,
                                      "kernel_size": 3, "padding": 1}},
    ]


def test_pyramid_emits_requested_strides_only() -> None:
    encoder = MultiScaleSemanticEncoder(out_channels=32, output_strides=[4, 16],
                                        norm_groups=8, pretrained=False)
    pyramid = encoder(torch.rand(2, 1, 128, 128))

    assert sorted(pyramid) == [4, 16]
    assert pyramid[4].shape == (2, 32, 32, 32)
    assert pyramid[16].shape == (2, 32, 8, 8)


def test_upsampled_strides_are_available() -> None:
    """The backbone stops at 1/4; a UNet decoder does not."""
    encoder = MultiScaleSemanticEncoder(out_channels=32, output_strides=[1, 2],
                                        norm_groups=8, pretrained=False)
    pyramid = encoder(torch.rand(1, 1, 64, 64))

    assert pyramid[1].shape == (1, 32, 64, 64)
    assert pyramid[2].shape == (1, 32, 32, 32)


def test_unknown_stride_rejected() -> None:
    with pytest.raises(ValueError, match="neither backbone strides"):
        MultiScaleSemanticEncoder(output_strides=[3], pretrained=False)


def test_norm_groups_must_divide_width() -> None:
    with pytest.raises(ValueError, match="must divide"):
        MultiScaleSemanticEncoder(out_channels=100, norm_groups=8, pretrained=False)


def test_backbone_stays_frozen_and_in_eval() -> None:
    encoder = MultiScaleSemanticEncoder(out_channels=32, pretrained=False)
    encoder.train()

    assert not any(parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert not encoder.backbone._features.training
    assert all(parameter.requires_grad for parameter in encoder.lateral_projections.parameters())


@pytest.mark.parametrize("variant,channels", [("tiny", [96, 192, 384, 768]),
                                              ("small", [96, 192, 384, 768]),
                                              ("base", [128, 256, 512, 1024])])
def test_backbone_variants(variant: str, channels: List[int]) -> None:
    backbone = FrozenConvNextBackbone(pretrained=False, variant=variant)
    assert backbone.getStageChannels() == channels

    stages = backbone(torch.rand(1, 1, 64, 64))
    assert [stage.shape[1] for stage in stages] == channels


def test_unknown_variant_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported ConvNeXt variant"):
        FrozenConvNextBackbone(pretrained=False, variant="huge")


def test_decoder_blocks_receive_their_own_resolution() -> None:
    """dec0 runs at 1/8 and dec1 at 1/4, and each must get the map of its own size."""
    network = UNetWithSkipConnections(
        buildLayers({"dec0": 32, "dec1": 32}),
        multi_scale_semantic_encoder={"out_channels": 32, "output_strides": [4, 8],
                                      "norm_groups": 8, "pretrained": False},
        semantic_input_channel=0
    )

    seen: Dict[str, Any] = {}
    for name in ("dec0", "dec1"):
        block = network.decoder_blocks[name]
        original = block.forward

        def wrapped(x, encoder_features, semantic_features=None, _name=name,
                    _original=original):
            seen[_name] = None if semantic_features is None else tuple(
                semantic_features.shape[-2:])
            return _original(x, encoder_features, semantic_features)

        block.forward = wrapped

    network(torch.rand(1, 1, 64, 64))

    assert seen == {"dec0": (8, 8), "dec1": (16, 16)}


def test_missing_stride_names_the_block() -> None:
    network = UNetWithSkipConnections(
        buildLayers({"dec3": 32}),
        multi_scale_semantic_encoder={"out_channels": 32, "output_strides": [4, 8],
                                      "norm_groups": 8, "pretrained": False},
        semantic_input_channel=0
    )

    with pytest.raises(ValueError, match="decoder block 'dec3' needs a semantic map"):
        network(torch.rand(1, 1, 64, 64))


def test_block_without_semantics_is_unchanged() -> None:
    """semantic_channels = 0 must leave the block bit-identical, so old checkpoints load."""
    plain = UNetWithSkipConnections(buildLayers({}))
    state = plain.state_dict()

    assert not any("semantic_merge" in key for key in state)
    assert plain(torch.rand(1, 1, 64, 64)).shape == (1, 2, 64, 64)


def test_two_semantic_encoders_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        UNetWithSkipConnections(
            buildLayers({}),
            semantic_encoder={"out_channels": 32, "norm_groups": 8, "pretrained": False},
            multi_scale_semantic_encoder={"out_channels": 32, "norm_groups": 8,
                                          "pretrained": False}
        )


def test_luminance_encoder_variant_selected_by_name() -> None:
    """The architecture file names the backbone, so a checkpoint states what produced it."""
    from pyaiwrap.neural_network import ColorMemoryTransformer

    model = ColorMemoryTransformer(pretrained_encoder="convnext_small",
                                   pretrained_encoder_stage_channels=[96, 256, 256, 512])
    assert model.luminance_encoder.backbone._variant == "small"


def test_unknown_pretrained_encoder_names_the_options() -> None:
    from pyaiwrap.neural_network import ColorMemoryTransformer

    with pytest.raises(ValueError, match="convnext_base"):
        ColorMemoryTransformer(pretrained_encoder="resnet50")
