"""Tests for the legacy explicit aspect-ratio band of the paired geometric augmentation.

The band randomises the crop's aspect and so distorts training differently from
validation; it is no longer the default (see test_aspect_preserving_crop.py) and survives
only to reproduce runs configured with it, where it was 0.85-1.18. These tests pin that
path so those configs keep behaving as they did when they were trained.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Tuple

from PIL import Image
from torchvision import transforms

from pyaiwrap.transforms import createSharedGeometricAugmentation


def _findRandomResizedCrop(composed: transforms.Compose) -> transforms.RandomResizedCrop:
    for transform in composed.transforms:
        if isinstance(transform, transforms.RandomResizedCrop):
            return transform
    raise AssertionError("no RandomResizedCrop in the composed augmentation")


def testRatioBandIsConfigurable() -> None:
    crop: transforms.RandomResizedCrop = _findRandomResizedCrop(
        createSharedGeometricAugmentation(256, ratio_min=0.7, ratio_max=1.5)
    )
    assert tuple(crop.ratio) == (0.7, 1.5)


def testScaleBandStillApplies() -> None:
    """The legacy path must keep honouring AUG_CROP_SCALE_MIN alongside the ratio band."""
    crop: transforms.RandomResizedCrop = _findRandomResizedCrop(
        createSharedGeometricAugmentation(256, crop_scale_min=0.4, ratio_min=0.85, ratio_max=1.18)
    )
    assert tuple(crop.scale) == (0.4, 1.0)


def testOutputIsAlwaysTheRequestedSquare() -> None:
    """Whatever the band, the augmentation emits IMAGE_RESIZE x IMAGE_RESIZE."""
    augmentation: transforms.Compose = createSharedGeometricAugmentation(
        128, crop_scale_min=0.4, ratio_min=0.7, ratio_max=1.5
    )
    for size in [(1200, 290), (400, 400), (300, 900)]:
        image: Image.Image = Image.new("RGB", size)
        for _ in range(5):
            assert augmentation(image).size == (128, 128)


def testWiderBandProducesMoreStretchedCrops() -> None:
    """The wider band must actually reach aspect ratios the narrow one cannot."""
    image: Image.Image = Image.new("RGB", (900, 900))

    def sampledRatios(ratio_band: Tuple[float, float]) -> list:
        crop: transforms.RandomResizedCrop = _findRandomResizedCrop(
            createSharedGeometricAugmentation(256, ratio_min=ratio_band[0], ratio_max=ratio_band[1])
        )
        ratios: list = []
        for _ in range(200):
            _, _, height, width = crop.get_params(image, list(crop.scale), list(crop.ratio))
            ratios.append(width / height)
        return ratios

    narrow: list = sampledRatios((0.85, 1.18))
    wide: list = sampledRatios((0.7, 1.5))

    assert max(wide) > max(narrow)
    assert min(wide) < min(narrow)
    # Every sample still respects its own band (small tolerance for integer rounding).
    assert all(0.83 <= r <= 1.20 for r in narrow)
    assert all(0.68 <= r <= 1.53 for r in wide)
