"""The inference-side output corrections: chroma scaling, flip TTA and the L* assembly."""
from __future__ import annotations

import kornia
import pytest
import torch
import torch.nn as nn

from pyaiwrap.inference import colorize, colorizeFromConfig, predictChroma


class ConstantChromaGenerator(nn.Module):
    """Predicts a fixed ab everywhere, so scaling and averaging are exactly checkable."""

    def __init__(self, a_value: float = 20.0, b_value: float = -10.0) -> None:
        super().__init__()
        self.a_value = a_value
        self.b_value = b_value

    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = luminance.shape
        a_channel = torch.full((batch, 1, height, width), self.a_value)
        b_channel = torch.full((batch, 1, height, width), self.b_value)
        return torch.cat([a_channel, b_channel], dim=1)


class LeftHalfGenerator(nn.Module):
    """Paints only the left half, so a horizontal flip changes the answer."""

    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = luminance.shape
        chroma = torch.zeros(batch, 2, height, width)
        chroma[:, :, :, : width // 2] = 40.0
        return chroma


def test_chroma_scale_scales_predicted_chroma() -> None:
    generator = ConstantChromaGenerator()
    luminance = torch.full((1, 1, 4, 4), 0.5)

    full = colorize(generator, luminance, "AB", chroma_scale=1.0)
    half = colorize(generator, luminance, "AB", chroma_scale=0.5)

    full_ab = kornia.color.rgb_to_lab(full)[:, 1:3]
    half_ab = kornia.color.rgb_to_lab(half)[:, 1:3]

    assert torch.allclose(half_ab, full_ab * 0.5, atol=0.5)


def test_zero_chroma_scale_is_neutral() -> None:
    generator = ConstantChromaGenerator()
    luminance = torch.full((1, 1, 4, 4), 0.5)

    result = colorize(generator, luminance, "AB", chroma_scale=0.0)

    assert torch.allclose(result[:, 0], result[:, 1], atol=1e-3)
    assert torch.allclose(result[:, 1], result[:, 2], atol=1e-3)


def test_flip_tta_averages_the_two_views() -> None:
    generator = LeftHalfGenerator()
    luminance = torch.full((1, 1, 4, 4), 0.5)

    plain = predictChroma(generator, luminance, "AB", flip_tta=False)
    averaged = predictChroma(generator, luminance, "AB", flip_tta=True)

    # Left half painted in the original view, right half in the mirrored one: averaging puts
    # half the chroma everywhere.
    assert torch.allclose(plain[:, :, :, :2], torch.full((1, 2, 4, 2), 40.0))
    assert torch.allclose(plain[:, :, :, 2:], torch.zeros(1, 2, 4, 2))
    assert torch.allclose(averaged, torch.full((1, 2, 4, 4), 20.0))


def test_flip_tta_is_identity_for_a_flip_equivariant_model() -> None:
    generator = ConstantChromaGenerator()
    luminance = torch.rand(2, 1, 8, 8)

    assert torch.allclose(predictChroma(generator, luminance, "AB", flip_tta=True),
                          predictChroma(generator, luminance, "AB", flip_tta=False))


def test_lightness_comes_from_the_input_not_the_model() -> None:
    generator = ConstantChromaGenerator(a_value=0.0, b_value=0.0)
    luminance = torch.tensor([[[[0.2, 0.5, 0.8]]]])

    result = colorize(generator, luminance, "AB", chroma_scale=1.0)
    lightness = kornia.color.rgb_to_lab(result)[:, 0:1]

    expected = kornia.color.rgb_to_lab(luminance.repeat(1, 3, 1, 1))[:, 0:1]
    assert torch.allclose(lightness, expected, atol=0.1)


def test_rgb_model_untouched_when_nothing_is_requested() -> None:
    """An RGB-output model pays for the Lab round trip, so it must be skipped by default."""

    class RgbGenerator(nn.Module):
        def forward(self, luminance: torch.Tensor) -> torch.Tensor:
            return torch.cat([luminance * 0.9, luminance, luminance * 1.1], dim=1)

    generator = RgbGenerator()
    luminance = torch.rand(1, 1, 4, 4) * 0.8

    direct = generator(luminance).clamp(0.0, 1.0)
    assert torch.equal(colorize(generator, luminance, "RGB"), direct)


def test_rgb_model_tta_stays_in_rgb() -> None:
    """An RGB model must not pay the Lab round trip just to get flip averaging.

    Doing it the other way round is what made rgb_merge_unet_v6 2.9% worse with TTA on: the
    round trip costs more than the averaging buys. The check is that the result is exactly the
    mean of the two RGB predictions, which a Lab detour would not reproduce.
    """

    class AsymmetricRgbGenerator(nn.Module):
        def forward(self, luminance: torch.Tensor) -> torch.Tensor:
            batch, _, height, width = luminance.shape
            image = torch.full((batch, 3, height, width), 0.5)
            image[:, 0, :, : width // 2] = 0.9
            return image

    generator = AsymmetricRgbGenerator()
    luminance = torch.full((1, 1, 4, 4), 0.5)

    plain = generator(luminance)
    expected = 0.5 * (plain + torch.flip(generator(torch.flip(luminance, dims=[-1])),
                                         dims=[-1]))

    assert torch.allclose(colorize(generator, luminance, "RGB", flip_tta=True), expected)


def test_config_defaults_are_the_identity() -> None:
    generator = ConstantChromaGenerator()
    luminance = torch.rand(1, 1, 4, 4)

    assert torch.allclose(colorizeFromConfig(generator, luminance, {"TARGET_CHANNEL": "AB"}),
                          colorize(generator, luminance, "AB"))


def test_unsupported_target_channel_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported target_channel"):
        predictChroma(ConstantChromaGenerator(), torch.rand(1, 1, 4, 4), "LAB")
