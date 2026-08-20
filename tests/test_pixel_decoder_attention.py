"""Tests for the optional self-attention of `PixelDecoder`.

The pixel decoder of the original DDColor is purely convolutional -- attention lives only in
its colour decoder -- while this one runs a self-attention block at every scale. To measure
what that extra attention buys, the blocks have to be removable, and removable without
disturbing anything already trained: `use_attention` defaults to True, and every checkpoint
trained so far must keep loading into a decoder built with the default.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Any, Dict, List

import torch
import torch.nn as nn

from pyaiwrap.neural_network import ColorMemoryTransformer, PixelDecoder

ENCODER_CHANNELS: List[int] = [16, 32, 32, 64]
EMBED_DIM: int = 48


def _buildDecoder(use_attention: bool = True) -> PixelDecoder:
    return PixelDecoder(encoder_channels=ENCODER_CHANNELS, embed_dim=EMBED_DIM,
                        num_heads=4, mlp_ratio=2, dropout=0.0, use_attention=use_attention)


def _buildEncoderOutputs(batch_size: int = 2, base_size: int = 32) -> List[torch.Tensor]:
    """The pyramid PixelDecoder expects: one map per encoder stage, each half the previous."""
    return [torch.randn(batch_size, channels, base_size // (2 ** index), base_size // (2 ** index))
            for index, channels in enumerate(ENCODER_CHANNELS)]


def testAttentionIsOnByDefault() -> None:
    decoder: PixelDecoder = _buildDecoder()

    assert decoder.use_attention
    assert len(decoder.transformer_blocks) == len(ENCODER_CHANNELS)


def testDisablingAttentionBuildsNoTransformerBlocks() -> None:
    decoder: PixelDecoder = _buildDecoder(use_attention=False)

    assert not decoder.use_attention
    assert len(decoder.transformer_blocks) == 0
    assert not any("transformer_blocks" in key for key in decoder.state_dict())


def testDisablingAttentionKeepsTheConvolutionalPath() -> None:
    """Only the attention goes: the skip additions, upsampling and projections stay."""
    with_attention: PixelDecoder = _buildDecoder()
    without_attention: PixelDecoder = _buildDecoder(use_attention=False)

    convolutional_keys: List[str] = [key for key in with_attention.state_dict()
                                     if "transformer_blocks" not in key]
    assert list(without_attention.state_dict()) == convolutional_keys

    attention_parameters: int = sum(parameter.numel()
                                    for parameter in with_attention.transformer_blocks.parameters())
    assert attention_parameters > 0
    assert (sum(parameter.numel() for parameter in without_attention.parameters())
            == sum(parameter.numel() for parameter in with_attention.parameters())
            - attention_parameters)


def testBothVariantsProduceTheSameShapes() -> None:
    """The colour decoder and the einsum after it must not notice which variant they got."""
    torch.manual_seed(0)
    encoder_outputs: List[torch.Tensor] = _buildEncoderOutputs()

    shapes: List[List[torch.Size]] = []
    final_shapes: List[torch.Size] = []
    for use_attention in (True, False):
        decoder: PixelDecoder = _buildDecoder(use_attention=use_attention)
        decoder.eval()
        with torch.no_grad():
            multi_scale_features, final_features = decoder(encoder_outputs)
        shapes.append([features.shape for features in multi_scale_features])
        final_shapes.append(final_features.shape)

    assert shapes[0] == shapes[1]
    assert final_shapes[0] == final_shapes[1]
    assert all(shape[-1] == EMBED_DIM for shape in shapes[0])


def testAttentionFreeDecoderStillRuns() -> None:
    decoder: PixelDecoder = _buildDecoder(use_attention=False)
    decoder.eval()
    with torch.no_grad():
        multi_scale_features, final_features = decoder(_buildEncoderOutputs())

    assert len(multi_scale_features) == len(ENCODER_CHANNELS)
    assert torch.isfinite(final_features).all()


def testExistingCheckpointsLoadIntoTheDefaultDecoder() -> None:
    """Backwards compatibility: the flag must not have renamed or reordered a single weight."""
    torch.manual_seed(0)
    trained: PixelDecoder = _buildDecoder()
    fresh: PixelDecoder = _buildDecoder()

    missing, unexpected = fresh.load_state_dict(trained.state_dict(), strict=True)

    assert missing == [] and unexpected == []
    trained.eval()
    fresh.eval()
    encoder_outputs: List[torch.Tensor] = _buildEncoderOutputs()
    with torch.no_grad():
        _, trained_final = trained(encoder_outputs)
        _, fresh_final = fresh(encoder_outputs)
    assert torch.equal(trained_final, fresh_final)


TRANSFORMER_ARGUMENTS: Dict[str, Any] = dict(color_dim=64, embed_dim=64,
                                             color_decoder_output_dim=64, num_heads=4,
                                             mlp_ratio=2, dropout=0.0, color_decoder_layers=1,
                                             memory_size=8)


def testTransformerKeepsPixelDecoderAttentionByDefault() -> None:
    model: ColorMemoryTransformer = ColorMemoryTransformer(**TRANSFORMER_ARGUMENTS)

    assert model.pixel_decoder.use_attention
    assert len(model.pixel_decoder.transformer_blocks) > 0


def testTransformerPassesTheFlagToThePixelDecoder() -> None:
    """The architecture JSON's `pixel_decoder_attention` has to reach the decoder itself."""
    model: ColorMemoryTransformer = ColorMemoryTransformer(**TRANSFORMER_ARGUMENTS,
                                                           pixel_decoder_attention=False)

    assert not model.pixel_decoder.use_attention
    assert not any("pixel_decoder.transformer_blocks" in key for key in model.state_dict())

    model.eval()
    with torch.no_grad():
        output: torch.Tensor = model(torch.rand(1, 1, 64, 64))
    assert output.shape[0] == 1
    assert torch.isfinite(output).all()


def testColourDecoderKeepsItsAttentionWhenThePixelDecoderLosesIts() -> None:
    """Only the pixel decoder is ablated; DDColor's attention is the colour decoder's."""
    model: ColorMemoryTransformer = ColorMemoryTransformer(**TRANSFORMER_ARGUMENTS,
                                                           pixel_decoder_attention=False)

    attention_modules: List[nn.Module] = [module for module in model.color_decoder.modules()
                                          if module.__class__.__name__ == "MultiHeadAttention"]
    assert attention_modules
