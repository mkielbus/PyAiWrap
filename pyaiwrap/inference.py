"""Turning a trained colorization generator's output into an image, at inference time.

The training loop never needs this: it compares tensors in whatever space the target lives in.
Everything downstream -- the quality gate, visualisation, deployment -- needs one agreed answer
to "what RGB does this model produce for this grey image", and that answer carries three
choices the loss does not make:

  * how the [0,1] luminance becomes L* (see transforms.grayToLightness);
  * whether predicted chroma is scaled before display, which is worth measuring per model
    because an over-confident model gives away LPIPS it can have back for free;
  * whether to average the prediction over a flip of the input, the cheapest variance
    reduction available for a model that is already trained.

Keeping them here means the gate and the deployment path cannot drift apart, which is how the
`luminance * 100` substitution survived three generations of models.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .transforms import labToRgb, luminanceToLabRange


def predictChroma(generator: nn.Module, luminance: torch.Tensor, target_channel: str,
                  flip_tta: bool = False) -> torch.Tensor:
    """Run the generator and return its prediction as Lab ab, whatever it natively outputs.

    flip_tta averages the prediction for the input and for its horizontal mirror (unflipped
    again). Colour is not a chiral property, so the two are answers to the same question and
    their average cancels part of the model's own noise. It costs one extra forward pass.
    """
    prediction = _predictChromaOnce(generator, luminance, target_channel)
    if not flip_tta:
        return prediction

    flipped = _predictChromaOnce(generator, torch.flip(luminance, dims=[-1]), target_channel)
    return 0.5 * (prediction + torch.flip(flipped, dims=[-1]))


def _predictChromaOnce(generator: nn.Module, luminance: torch.Tensor,
                       target_channel: str) -> torch.Tensor:
    prediction = generator(luminance).float()
    if target_channel == "AB":
        return prediction
    if target_channel == "RGB":
        import kornia
        return kornia.color.rgb_to_lab(prediction.clamp(0.0, 1.0))[:, 1:3]
    raise ValueError(f"unsupported target_channel {target_channel!r}, expected 'AB' or 'RGB'")


def colorize(generator: nn.Module, luminance: torch.Tensor, target_channel: str,
             chroma_scale: float = 1.0, flip_tta: bool = False,
             luminance_transfer: str = "srgb") -> torch.Tensor:
    """The model's RGB output for a [B, 1, H, W] luminance input in [0, 1].

    A segmentation-conditioned model takes [B, 1 + C, H, W] instead, luminance first and the
    encoded label map behind it; everything here works on that stack unchanged, and flip TTA
    mirrors the conditioning along with the image, as it must.

    For an RGB-output model with chroma_scale == 1 and no TTA this is not bit-identical to
    calling the generator directly: the prediction is routed through Lab so that the returned
    image's lightness comes from the input rather than from the model, which is the property
    the gate and the deployment path both want. Measured on rgb_merge_unet_v6 that routing
    costs a little (0.1310 -> 0.1379 raw LPIPS), so RGB models are returned untouched unless a
    chroma adjustment is actually requested.
    """
    if target_channel == "RGB" and chroma_scale == 1.0:
        # Averaged in RGB, never through Lab. Routing an RGB model's output through ab and
        # back costs it about 5% of raw LPIPS (measured on rgb_merge_unet_v6: 0.1310 -> 0.1379),
        # which is more than flip TTA is worth -- doing it the other way round made the model
        # 2.9% worse rather than better.
        prediction = generator(luminance).float()
        if flip_tta:
            flipped = generator(torch.flip(luminance, dims=[-1])).float()
            prediction = 0.5 * (prediction + torch.flip(flipped, dims=[-1]))
        return prediction.clamp(0.0, 1.0)

    chroma = predictChroma(generator, luminance, target_channel, flip_tta=flip_tta)
    if chroma_scale != 1.0:
        chroma = chroma * chroma_scale

    # Channel 0 only: a segmentation-conditioned model is handed [luminance, mask encoding],
    # and the lightness the output is rebuilt from is the photograph's, not the conditioning's.
    # For the plain 1-channel case this slice is a no-op.
    lightness = luminanceToLabRange(luminance[:, :1], luminance_transfer)
    return labToRgb(lightness, chroma).clamp(0.0, 1.0)


def colorizeFromConfig(generator: nn.Module, luminance: torch.Tensor,
                       config: Optional[dict] = None) -> torch.Tensor:
    """colorize() with its three choices read from a training config."""
    config = config or {}
    return colorize(
        generator=generator,
        luminance=luminance,
        target_channel=config.get("TARGET_CHANNEL", "AB"),
        chroma_scale=config.get("INFERENCE_CHROMA_SCALE", 1.0),
        flip_tta=config.get("INFERENCE_FLIP_TTA", False),
        luminance_transfer=config.get("LUMINANCE_TRANSFER", "srgb"),
    )
