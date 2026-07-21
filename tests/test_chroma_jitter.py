"""Tests for ChromaJitter (Phase 1b L5a: BigColor-style chroma scaling).

ChromaJitter scales LAB chroma of the target image (ab <- s*ab, no hue change),
bounded so the resulting mean chroma stays inside the dataset's empirical
[p2, p98] band. Chroma is measured in OpenCV LAB units, matching
analysis/extract_colors.py, so the band bounds apply directly.

Naming/style follows the project convention (see CLAUDE.md).
"""
import random

import cv2
import numpy as np
import pytest
from PIL import Image

from pyaiwrap.transforms import ChromaJitter

IMAGE_SIZE: int = 64


def _measureMeanChroma(image: Image.Image) -> float:
    """OpenCV-LAB mean chroma, identical definition to extract_colors.py."""
    rgb: np.ndarray = np.asarray(image.convert("RGB"))
    lab: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a: np.ndarray = lab[..., 1] - 128.0
    b: np.ndarray = lab[..., 2] - 128.0
    return float(np.sqrt(a * a + b * b).mean())


def _measureMeanLightness(image: Image.Image) -> float:
    rgb: np.ndarray = np.asarray(image.convert("RGB"))
    lab: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(lab[..., 0].mean())


def _solidImage(rgb_tuple: tuple) -> Image.Image:
    pixels: np.ndarray = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    pixels[:, :] = rgb_tuple
    return Image.fromarray(pixels, mode="RGB")


def testProbabilityZeroIsIdentity() -> None:
    image: Image.Image = _solidImage((200, 120, 60))
    jitter: ChromaJitter = ChromaJitter(probability=0.0)
    result: Image.Image = jitter(image)
    assert np.array_equal(np.asarray(result), np.asarray(image))


def testScaleUpIncreasesChromaAndKeepsLightness() -> None:
    image: Image.Image = _solidImage((190, 130, 90))          # moderate chroma, inside band
    before_chroma: float = _measureMeanChroma(image)
    before_light: float = _measureMeanLightness(image)
    # fix s = 1.3 (min == max), band wide enough not to clamp
    jitter: ChromaJitter = ChromaJitter(probability=1.0, chroma_min=1.3, chroma_max=1.3,
                                        chroma_band_low=0.0, chroma_band_high=1e6)
    result: Image.Image = jitter(image)
    assert _measureMeanChroma(result) > before_chroma + 1.0
    assert abs(_measureMeanLightness(result) - before_light) < 2.0   # L preserved (uint8 roundtrip)


def testScaleDownDecreasesChroma() -> None:
    image: Image.Image = _solidImage((190, 130, 90))
    before_chroma: float = _measureMeanChroma(image)
    jitter: ChromaJitter = ChromaJitter(probability=1.0, chroma_min=0.6, chroma_max=0.6,
                                        chroma_band_low=0.0, chroma_band_high=1e6)
    assert _measureMeanChroma(jitter(image)) < before_chroma - 1.0


def testAchromaticIsPreserved() -> None:
    image: Image.Image = _solidImage((128, 128, 128))          # gray: chroma ~ 0
    jitter: ChromaJitter = ChromaJitter(probability=1.0, chroma_min=1.5, chroma_max=1.5,
                                        chroma_band_low=0.0, chroma_band_high=1e6)
    result: Image.Image = jitter(image)
    assert _measureMeanChroma(result) < 2.0
    assert np.abs(np.asarray(result).astype(int) - np.asarray(image).astype(int)).max() <= 3


def testUpperBandIsNotExceeded() -> None:
    """An already-saturated image must not be pushed above the band."""
    image: Image.Image = _solidImage((230, 40, 40))            # high chroma
    before_chroma: float = _measureMeanChroma(image)
    # band_high below the image's chroma; an upscale would exceed it -> must be refused
    jitter: ChromaJitter = ChromaJitter(probability=1.0, chroma_min=1.5, chroma_max=1.5,
                                        chroma_band_low=0.0, chroma_band_high=5.0)
    assert _measureMeanChroma(jitter(image)) <= before_chroma + 1.0


def testDesaturatedImageIsPushedTowardTheBand() -> None:
    """A very desaturated image is scaled up (toward, not beyond, the band)."""
    image: Image.Image = _solidImage((140, 132, 124))          # low chroma
    before_chroma: float = _measureMeanChroma(image)
    jitter: ChromaJitter = ChromaJitter(probability=1.0, chroma_min=0.9, chroma_max=1.25,
                                        chroma_band_low=30.0, chroma_band_high=43.0)
    random.seed(0)
    assert _measureMeanChroma(jitter(image)) >= before_chroma


def testRejectsNonPilInput() -> None:
    jitter: ChromaJitter = ChromaJitter()
    with pytest.raises(TypeError):
        jitter(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
