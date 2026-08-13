"""Tests for replaying one geometric draw on both the photograph and its label map.

A misaligned mask is worse than no mask: it teaches the network that object outlines sit
somewhere other than where they do, and the error is invisible in the loss curve. So the
invariants pinned here are that the flip and the crop box are shared, that the box transfers
correctly to a raster stored at a different resolution (SAM writes 256x256 maps whatever the
scan's size was), and that the mask is never interpolated into label ids that do not exist.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import List, Set

import numpy as np
import pytest
from PIL import Image

from pyaiwrap.transforms import (CropBox, GeometricAugmentationParameters,
                                 PairedGeometricAugmentation, createPairedGeometricAugmentation)

BLOCK_LABELS: int = 4


def _blockLabelMap(size: int) -> np.ndarray:
    """size x size raster split into BLOCK_LABELS vertical bands labelled 1..BLOCK_LABELS."""
    band: int = size // BLOCK_LABELS
    label_map: np.ndarray = np.zeros((size, size), dtype=np.uint8)
    for index in range(BLOCK_LABELS):
        label_map[:, index * band:(index + 1) * band] = index + 1
    return label_map


def _maskImage(size: int) -> Image.Image:
    return Image.fromarray(_blockLabelMap(size), mode="L")


def _imageMatchingMask(size: int) -> Image.Image:
    """The same bands as the label map, painted as grey levels 40, 80, 120, 160.

    Bands are wide, so bilinear resampling reproduces each band's grey exactly except within
    a pixel of the seams -- which makes "did the image and the mask receive the same box"
    checkable by comparing the band each pixel landed in.
    """
    label_map: np.ndarray = _blockLabelMap(size)
    return Image.fromarray((label_map * 40).astype(np.uint8), mode="L")


def _bandAgreement(augmented_image: Image.Image, augmented_mask: Image.Image) -> float:
    """Fraction of pixels whose grey level names the same band as the label under it."""
    image_bands: np.ndarray = np.rint(np.asarray(augmented_image, dtype=np.float32) / 40.0)
    mask_bands: np.ndarray = np.asarray(augmented_mask, dtype=np.float32)
    return float((image_bands == mask_bands).mean())


def testCropBoxRescalesProportionally() -> None:
    box: CropBox = CropBox(top=10, left=20, height=40, width=60, source_height=100, source_width=200)
    rescaled: CropBox = box.rescaleTo(50, 100)

    assert (rescaled.top, rescaled.left) == (5, 10)
    assert (rescaled.height, rescaled.width) == (20, 30)
    assert (rescaled.source_height, rescaled.source_width) == (50, 100)


def testCropBoxStaysInsideTheTargetFrame() -> None:
    """Rounding must never produce a rectangle that runs off the edge of a small raster."""
    box: CropBox = CropBox(top=99, left=99, height=1, width=1, source_height=100, source_width=100)
    rescaled: CropBox = box.rescaleTo(7, 7)

    assert rescaled.top + rescaled.height <= 7
    assert rescaled.left + rescaled.width <= 7
    assert rescaled.height >= 1 and rescaled.width >= 1


def testImageAndMaskReceiveTheSameCrop() -> None:
    augmentation: PairedGeometricAugmentation = createPairedGeometricAugmentation(
        image_size=64, flip_probability=0.5, crop_scale_min=0.4
    )
    image: Image.Image = _imageMatchingMask(64)
    mask: Image.Image = _maskImage(64)

    for _ in range(20):
        parameters: GeometricAugmentationParameters = augmentation.sampleParameters(64, 64)
        agreement: float = _bandAgreement(
            augmentation.applyToImage(image, parameters),
            augmentation.applyToMask(mask, parameters)
        )
        assert agreement > 0.9


def testMaskAtADifferentResolutionStaysAligned() -> None:
    """The photograph is whatever size the scan was; the label map is always 256x256."""
    augmentation: PairedGeometricAugmentation = createPairedGeometricAugmentation(
        image_size=64, crop_scale_min=0.4
    )
    image: Image.Image = _imageMatchingMask(128)
    mask: Image.Image = _maskImage(64)

    for _ in range(20):
        parameters: GeometricAugmentationParameters = augmentation.sampleParameters(128, 128)
        agreement: float = _bandAgreement(
            augmentation.applyToImage(image, parameters),
            augmentation.applyToMask(mask, parameters)
        )
        assert agreement > 0.9


def testFlipIsShared() -> None:
    augmentation: PairedGeometricAugmentation = createPairedGeometricAugmentation(
        image_size=64, crop_scale_min=1.0
    )
    parameters: GeometricAugmentationParameters = GeometricAugmentationParameters(
        flip=True, crop_box=CropBox(0, 0, 64, 64, 64, 64)
    )

    flipped_mask: np.ndarray = np.asarray(augmentation.applyToMask(_maskImage(64), parameters))
    assert flipped_mask[0, 0] == BLOCK_LABELS      # the last band is now first
    assert flipped_mask[0, -1] == 1

    agreement: float = _bandAgreement(
        augmentation.applyToImage(_imageMatchingMask(64), parameters), Image.fromarray(flipped_mask)
    )
    assert agreement > 0.9


def testMaskResamplingNeverInventsLabels() -> None:
    """NEAREST throughout: interpolating region 1 and region 3 would create a region 2."""
    augmentation: PairedGeometricAugmentation = createPairedGeometricAugmentation(
        image_size=37, crop_scale_min=0.4          # odd size, so the resample is never a no-op
    )
    original: Set[int] = set(np.unique(_blockLabelMap(64)).tolist())

    for _ in range(20):
        parameters: GeometricAugmentationParameters = augmentation.sampleParameters(64, 64)
        augmented: np.ndarray = np.asarray(augmentation.applyToMask(_maskImage(64), parameters))
        assert set(np.unique(augmented).tolist()).issubset(original)


def testOutputIsAlwaysTheRequestedSquare() -> None:
    augmentation: PairedGeometricAugmentation = createPairedGeometricAugmentation(
        image_size=48, crop_scale_min=0.4
    )
    sizes: List[tuple] = [(200, 100), (100, 200), (64, 64)]

    for width, height in sizes:
        parameters: GeometricAugmentationParameters = augmentation.sampleParameters(width, height)
        assert augmentation.applyToImage(_imageMatchingMask(64).resize((width, height)),
                                         parameters).size == (48, 48)
        assert augmentation.applyToMask(_maskImage(64), parameters).size == (48, 48)


def testCropStaysInsideTheSourceFrame() -> None:
    augmentation: PairedGeometricAugmentation = createPairedGeometricAugmentation(
        image_size=64, crop_scale_min=0.1
    )

    for _ in range(50):
        box: CropBox = augmentation.sampleParameters(123, 77).crop_box
        assert 0 <= box.left and box.left + box.width <= 123
        assert 0 <= box.top and box.top + box.height <= 77


def testCropKeepsTheSourceAspectRatio() -> None:
    """Same reason as the composed transform: validation squares off the whole frame."""
    augmentation: PairedGeometricAugmentation = createPairedGeometricAugmentation(
        image_size=64, crop_scale_min=0.4
    )

    for _ in range(50):
        box: CropBox = augmentation.sampleParameters(200, 100).crop_box
        assert box.width / box.height == pytest.approx(2.0, rel=0.05)


def testFlipProbabilityIsHonoured() -> None:
    never: PairedGeometricAugmentation = createPairedGeometricAugmentation(64, flip_probability=0.0)
    always: PairedGeometricAugmentation = createPairedGeometricAugmentation(64, flip_probability=1.0)

    assert not any(never.sampleParameters(64, 64).flip for _ in range(20))
    assert all(always.sampleParameters(64, 64).flip for _ in range(20))


def testRejectsInvalidFlipProbability() -> None:
    with pytest.raises(ValueError, match="flip_probability"):
        createPairedGeometricAugmentation(64, flip_probability=1.5)


def testCallableFormMatchesTheComposedTransformContract() -> None:
    """It must still work as a plain image-only augmentation, for code that does not pair."""
    augmentation: PairedGeometricAugmentation = createPairedGeometricAugmentation(
        image_size=32, crop_scale_min=0.5
    )
    assert augmentation(_imageMatchingMask(64)).size == (32, 32)
