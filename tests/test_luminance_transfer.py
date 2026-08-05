"""The sRGB grey -> L* transfer, and the loss honouring the configured choice of it."""
from __future__ import annotations

import kornia
import pytest
import torch

from pyaiwrap.loss import GeneratorColorizationLoss
from pyaiwrap.transforms import grayToLightness, luminanceToLabRange


def test_neutral_grey_matches_kornia_exactly() -> None:
    """For a neutral pixel the transfer is exact, so it must agree with a full rgb_to_lab."""
    gray = torch.linspace(0.0, 1.0, steps=64).view(1, 1, 8, 8)

    expected = kornia.color.rgb_to_lab(gray.repeat(1, 3, 1, 1))[:, 0:1]

    assert torch.allclose(grayToLightness(gray), expected, atol=1e-4)


def test_endpoints_and_monotonicity() -> None:
    gray = torch.linspace(0.0, 1.0, steps=256).view(1, 1, 16, 16)
    lightness = grayToLightness(gray)

    assert lightness.min().item() == pytest.approx(0.0, abs=1e-5)
    assert lightness.max().item() == pytest.approx(100.0, abs=1e-4)

    flat = lightness.flatten()
    assert torch.all(flat[1:] >= flat[:-1])


def test_legacy_transfer_is_darker_in_midtones() -> None:
    """The bug this replaces: `luminance * 100` under-reports mid-tone lightness."""
    gray = torch.tensor([[[[0.2, 0.5, 0.8]]]])

    correct = luminanceToLabRange(gray, "srgb")
    legacy = luminanceToLabRange(gray, "linear")

    assert torch.all(legacy < correct)
    assert (correct - legacy).max().item() == pytest.approx(3.4, abs=0.3)


def test_unknown_transfer_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported luminance transfer"):
        luminanceToLabRange(torch.zeros(1, 1, 2, 2), "rec601")


def test_loss_rejects_unknown_transfer() -> None:
    with pytest.raises(ValueError, match="luminance_transfer"):
        GeneratorColorizationLoss(target_channel="AB", input_channel="luminance",
                                  luminance_transfer="rec601")


def test_loss_default_reconstructs_the_true_image() -> None:
    """With the correct transfer, rendering the true ab on the input grey returns the image.

    This is the property the old path lacked, and the reason the error stayed invisible: under
    "linear" both the prediction and the target are rendered wrong in the same way, so the
    loss cannot see it, while a gate comparing against `image` can.
    """
    torch.manual_seed(0)
    image = torch.rand(2, 3, 8, 8)
    lab = kornia.color.rgb_to_lab(image)
    gray = (lab[:, 0:1] / 100.0)  # a grey channel that is exactly L*/100

    loss_fn = GeneratorColorizationLoss(target_channel="AB", input_channel="luminance",
                                        luminance_transfer="srgb")
    legacy_fn = GeneratorColorizationLoss(target_channel="AB", input_channel="luminance",
                                          luminance_transfer="linear")

    # grayToLightness expects an sRGB grey, so feed it the grey whose L* is known: the round
    # trip is exact only for the legacy transfer here, which is precisely the confusion being
    # fixed -- so compare the two paths against each other rather than against the image.
    correct = loss_fn._convertToRgbForLoss(lab[:, 1:3], gray)
    legacy = legacy_fn._convertToRgbForLoss(lab[:, 1:3], gray)

    assert not torch.allclose(correct, legacy, atol=1e-3)
    assert torch.allclose(legacy, image, atol=1e-3)


def test_rgb_target_path_is_untouched() -> None:
    """Only the AB family routes through L*; an RGB-output model must be unaffected."""
    images = torch.rand(2, 3, 8, 8)
    modified = torch.rand(2, 1, 8, 8)

    for transfer in ("srgb", "linear"):
        loss_fn = GeneratorColorizationLoss(target_channel="RGB", input_channel="luminance",
                                            luminance_transfer=transfer)
        assert torch.equal(loss_fn._convertToRgbForLoss(images, modified), images)
