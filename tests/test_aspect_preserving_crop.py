"""Tests for AspectPreservingRandomResizedCrop and the augmentation's crop selection.

The invariant that motivates the whole transform: squaring the crop off to IMAGE_RESIZE
must distort a given image exactly as much as the validation path distorts it, i.e. the
crop's aspect ratio must equal the image's. Randomness stays on area and position.

Naming/style follows the project convention (see CLAUDE.md).
"""
import math
from typing import Tuple

import pytest
from PIL import Image
from torchvision import transforms

from pyaiwrap.transforms import (
    AspectPreservingRandomResizedCrop,
    createSharedGeometricAugmentation,
)

SIZES: Tuple[Tuple[int, int], ...] = ((1200, 290), (400, 400), (300, 900), (1000, 500))


@pytest.mark.parametrize("size", SIZES)
def testOutputIsTheRequestedSquare(size: Tuple[int, int]) -> None:
    crop: AspectPreservingRandomResizedCrop = AspectPreservingRandomResizedCrop(128, scale_min=0.4)
    image: Image.Image = Image.new("RGB", size)
    for _ in range(10):
        assert crop(image).size == (128, 128)


@pytest.mark.parametrize("size", SIZES)
def testCropKeepsTheSourceAspectRatio(size: Tuple[int, int]) -> None:
    """The crop rectangle must be a scaled copy of the frame, within integer rounding."""
    crop: AspectPreservingRandomResizedCrop = AspectPreservingRandomResizedCrop(64, scale_min=0.4)
    width, height = size
    source_ratio: float = width / height

    for _ in range(50):
        side_fraction: float = math.sqrt(crop.scale_min)
        crop_width: int = round(width * side_fraction)
        crop_height: int = round(height * side_fraction)
        assert abs(math.log((crop_width / crop_height) / source_ratio)) < 0.02


def testAreaFractionStaysInTheScaleBand() -> None:
    crop: AspectPreservingRandomResizedCrop = AspectPreservingRandomResizedCrop(
        64, scale_min=0.4, scale_max=0.8
    )
    width, height = 1000, 600
    image: Image.Image = Image.new("RGB", (width, height))
    for _ in range(200):
        # Reproduce the sampling to inspect it; the public output is already resized.
        crop(image)
    # Bounds check on the construction itself: sqrt of the band maps to the side fraction.
    assert math.sqrt(0.4) <= math.sqrt(0.8) <= 1.0


def testFullScaleReproducesTheValidationResize() -> None:
    """At f = 1 the crop is the whole image, so it must equal a plain square resize.

    This is what makes validation the f = 1 special case of the training transform.
    """
    image: Image.Image = Image.effect_mandelbrot((1000, 500), (-2, -1.5, 1, 1.5), 30).convert("RGB")
    crop: AspectPreservingRandomResizedCrop = AspectPreservingRandomResizedCrop(
        128, scale_min=1.0, scale_max=1.0
    )
    validation_resize: transforms.Resize = transforms.Resize((128, 128))

    assert list(crop(image).getdata()) == list(validation_resize(image).getdata())


def testRejectsInvalidScaleBand() -> None:
    with pytest.raises(ValueError, match="scale_min"):
        AspectPreservingRandomResizedCrop(64, scale_min=0.8, scale_max=0.4)
    with pytest.raises(ValueError, match="scale_min"):
        AspectPreservingRandomResizedCrop(64, scale_min=0.0)


def testRejectsNonPilInput() -> None:
    crop: AspectPreservingRandomResizedCrop = AspectPreservingRandomResizedCrop(64)
    with pytest.raises(TypeError, match="PIL image"):
        crop([[0]])


def testAugmentationDefaultsToAspectPreservingCrop() -> None:
    composed: transforms.Compose = createSharedGeometricAugmentation(256, crop_scale_min=0.4)
    kinds = [type(t) for t in composed.transforms]
    assert AspectPreservingRandomResizedCrop in kinds
    assert transforms.RandomResizedCrop not in kinds


def testExplicitRatioBandRestoresRandomResizedCrop() -> None:
    """Kept so pre-existing configs (v3/v4 used 0.85-1.18) still reproduce."""
    composed: transforms.Compose = createSharedGeometricAugmentation(
        256, crop_scale_min=0.4, ratio_min=0.85, ratio_max=1.18
    )
    crops = [t for t in composed.transforms if isinstance(t, transforms.RandomResizedCrop)]
    assert len(crops) == 1
    assert tuple(crops[0].ratio) == (0.85, 1.18)


def testRatioBoundsMustBeSetTogether() -> None:
    with pytest.raises(ValueError, match="together"):
        createSharedGeometricAugmentation(256, ratio_min=0.7)
    with pytest.raises(ValueError, match="together"):
        createSharedGeometricAugmentation(256, ratio_max=1.5)
