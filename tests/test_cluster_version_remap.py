"""Tests for ClusterVersionRemap (Phase 1b L5b: cluster-conditioned version remap).

The transform remaps whole color versions within a cluster: chromatic source colors
are hue-mapped band->band (relative position preserved) with S/V matching to the target
color, chromatic->achromatic is allowed as desaturation, and achromatic source pixels are
left untouched. Colors are classified with the exact bands from analysis/extract_colors.py.

Naming/style follows the project convention (see CLAUDE.md).
"""
import cv2
import numpy as np
import pytest
from PIL import Image

from pyaiwrap.transforms import ClusterVersionRemap, RemapTarget

SIZE: int = 64
SATURATION_THRESHOLD: float = 0.20
BROWN_HUE: tuple = (15.0, 50.0)
BROWN_V: float = 0.55
HUE_BINS: tuple = (
    ("red", 345.0, 15.0), ("orange", 15.0, 45.0), ("yellow", 45.0, 70.0),
    ("green", 70.0, 165.0), ("cyan", 165.0, 200.0), ("blue", 200.0, 255.0),
    ("purple", 255.0, 290.0), ("magenta", 290.0, 320.0), ("pink", 320.0, 345.0),
)


def _solidHsvImage(hue_deg: float, saturation: float, value: float, size: int = SIZE) -> Image.Image:
    hsv: np.ndarray = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = int(round(hue_deg / 2.0)) % 180
    hsv[..., 1] = int(round(saturation * 255.0))
    hsv[..., 2] = int(round(value * 255.0))
    rgb: np.ndarray = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb, mode="RGB")


def _dominantColor(image: Image.Image) -> str:
    rgb: np.ndarray = np.asarray(image.convert("RGB"))
    hsv: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue: np.ndarray = hsv[..., 0].astype(np.float32) * 2.0
    saturation: np.ndarray = hsv[..., 1].astype(np.float32) / 255.0
    value: np.ndarray = hsv[..., 2].astype(np.float32) / 255.0
    counts: dict = {}
    achromatic: np.ndarray = saturation < SATURATION_THRESHOLD
    counts["black"] = int(np.count_nonzero(achromatic & (value < 0.20)))
    counts["white"] = int(np.count_nonzero(achromatic & (value >= 0.85)))
    counts["gray"] = int(np.count_nonzero(achromatic) - counts["black"] - counts["white"])
    chromatic: np.ndarray = ~achromatic
    brown: np.ndarray = chromatic & (hue >= BROWN_HUE[0]) & (hue < BROWN_HUE[1]) & (value < BROWN_V)
    counts["brown"] = int(np.count_nonzero(brown))
    rest: np.ndarray = chromatic & ~brown
    for name, start, end in HUE_BINS:
        if start < end:
            in_bin = rest & (hue >= start) & (hue < end)
        else:
            in_bin = rest & ((hue >= start) | (hue < end))
        counts[name] = int(np.count_nonzero(in_bin))
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _meanValue(image: Image.Image) -> float:
    hsv: np.ndarray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    return float(hsv[..., 2].astype(np.float32).mean() / 255.0)


def _target(color: str, saturation_mean: float = 0.6, value_mean: float = 0.6) -> RemapTarget:
    return RemapTarget(target_color=color, saturation_mean=saturation_mean,
                       saturation_std=0.05, value_mean=value_mean, value_std=0.05)


def testProbabilityZeroIsIdentity() -> None:
    image: Image.Image = _solidHsvImage(227.0, 0.7, 0.7)          # blue
    remap: ClusterVersionRemap = ClusterVersionRemap({"blue": _target("green")}, probability=0.0)
    assert np.array_equal(np.asarray(remap(image)), np.asarray(image))


def testChromaticSourceIsHueMappedToTarget() -> None:
    image: Image.Image = _solidHsvImage(227.0, 0.7, 0.7)          # blue
    assert _dominantColor(image) == "blue"
    remap: ClusterVersionRemap = ClusterVersionRemap(
        {"blue": _target("green", value_mean=0.7)}, probability=1.0)
    assert _dominantColor(remap(image)) == "green"


def testChromaticToAchromaticDesaturates() -> None:
    image: Image.Image = _solidHsvImage(30.0, 0.7, 0.7)           # orange
    remap: ClusterVersionRemap = ClusterVersionRemap(
        {"orange": _target("gray", saturation_mean=0.05, value_mean=0.6)}, probability=1.0)
    result: Image.Image = remap(image)
    assert _dominantColor(result) in ("gray", "black", "white")


def testAchromaticPixelsAreUntouched() -> None:
    """A gray region must survive a remap targeting a chromatic color."""
    gray_rgb: np.ndarray = np.asarray(_solidHsvImage(0.0, 0.0, 0.5))       # sat 0 -> gray
    blue_rgb: np.ndarray = np.asarray(_solidHsvImage(227.0, 0.7, 0.7))     # blue
    composite: np.ndarray = blue_rgb.copy()
    composite[:, : SIZE // 2] = gray_rgb[:, : SIZE // 2]                    # left half gray
    image: Image.Image = Image.fromarray(composite, mode="RGB")

    remap: ClusterVersionRemap = ClusterVersionRemap(
        {"blue": _target("green", value_mean=0.7)}, probability=1.0)
    result: np.ndarray = np.asarray(remap(image))
    left_before: np.ndarray = composite[:, : SIZE // 2].astype(int)
    left_after: np.ndarray = result[:, : SIZE // 2].astype(int)
    assert np.abs(left_after - left_before).max() <= 3                     # gray untouched


def testSvMatchingDarkensYellowToBrown() -> None:
    image: Image.Image = _solidHsvImage(57.0, 0.7, 0.9)           # bright yellow
    remap: ClusterVersionRemap = ClusterVersionRemap(
        {"yellow": _target("brown", saturation_mean=0.6, value_mean=0.3)}, probability=1.0)
    result: Image.Image = remap(image)
    assert _dominantColor(result) == "brown"
    assert _meanValue(result) < 0.5                                        # darkened toward target


def testUnmappedChromaticColorIsUntouched() -> None:
    image: Image.Image = _solidHsvImage(120.0, 0.7, 0.7)          # green, not in correspondence
    remap: ClusterVersionRemap = ClusterVersionRemap({"blue": _target("red")}, probability=1.0)
    result: np.ndarray = np.asarray(remap(image))
    assert np.abs(result.astype(int) - np.asarray(image).astype(int)).max() <= 3


def testRejectsNonPilInput() -> None:
    remap: ClusterVersionRemap = ClusterVersionRemap({"blue": _target("green")})
    with pytest.raises(TypeError):
        remap(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))


def testDarkShadowsAreNotRecolored() -> None:
    """Shadows are near-black regions that keep their colour under any real recolour: a dark
    green shade must survive a green->red remap even though it is chromatic."""
    lit_rgb: np.ndarray = np.asarray(_solidHsvImage(120.0, 0.6, 0.7))       # lit green
    shadow_rgb: np.ndarray = np.asarray(_solidHsvImage(120.0, 0.6, 0.10))   # green in deep shade
    composite: np.ndarray = lit_rgb.copy()
    composite[:, : SIZE // 2] = shadow_rgb[:, : SIZE // 2]                  # left half in shadow
    image: Image.Image = Image.fromarray(composite, mode="RGB")

    remap: ClusterVersionRemap = ClusterVersionRemap(
        {"green": _target("red", value_mean=0.7)}, probability=1.0)
    result: np.ndarray = np.asarray(remap(image))
    left_before: np.ndarray = composite[:, : SIZE // 2].astype(int)
    left_after: np.ndarray = result[:, : SIZE // 2].astype(int)
    assert np.abs(left_after - left_before).max() <= 3                      # shadow untouched
    assert _dominantColor(Image.fromarray(result[:, SIZE // 2:], mode="RGB")) == "red"


def testShadowProtectionIsFeathered() -> None:
    """Just above the shadow floor the remap ramps in, so there is no hard seam."""
    remap: ClusterVersionRemap = ClusterVersionRemap(
        {"green": _target("red", value_mean=0.7)}, probability=1.0,
        shadow_value=0.25, feather_value=0.10)
    below: Image.Image = _solidHsvImage(120.0, 0.6, 0.20)     # fully protected
    partway: Image.Image = _solidHsvImage(120.0, 0.6, 0.30)   # half-way up the ramp
    above: Image.Image = _solidHsvImage(120.0, 0.6, 0.50)     # fully remapped

    def hueOf(image: Image.Image) -> float:
        hsv: np.ndarray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
        return float(hsv[..., 0].astype(np.float32).mean() * 2.0)

    assert hueOf(remap(below)) == pytest.approx(hueOf(below), abs=2.0)
    assert hueOf(remap(above)) < 60.0                                       # moved to red/orange
    assert hueOf(remap(above)) < hueOf(remap(partway)) < hueOf(below)       # monotone ramp


def _noisyHueImage(center_deg: float, spread_deg: float, saturation: float, value: float,
                   seed: int = 0, size: int = SIZE) -> Image.Image:
    """A flat surface whose hue jitters across a band edge -- the real speckle trigger."""
    rng: np.random.Generator = np.random.default_rng(seed)
    hue: np.ndarray = (center_deg + rng.uniform(-spread_deg, spread_deg, (size, size))) % 360.0
    hsv: np.ndarray = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = np.round(hue / 2.0).astype(np.uint8) % 180
    hsv[..., 1] = int(round(saturation * 255.0))
    hsv[..., 2] = int(round(value * 255.0))
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mode="RGB")


def _totalVariation(image: Image.Image) -> float:
    """Mean absolute difference between horizontally adjacent pixels: high = speckled."""
    rgb: np.ndarray = np.asarray(image.convert("RGB")).astype(np.float64)
    return float(np.abs(np.diff(rgb, axis=1)).mean())


def testWeightSmoothingSuppressesBandEdgeSpeckle() -> None:
    """A surface straddling the orange/yellow edge (45 deg) must not come out salt-and-pepper:
    per-pixel band membership is noise, so the blend weight is spatially smoothed."""
    image: Image.Image = _noisyHueImage(45.0, 8.0, saturation=0.5, value=0.6)
    correspondence: dict = {"orange": _target("green", value_mean=0.6)}
    speckled: Image.Image = ClusterVersionRemap(
        correspondence, probability=1.0, feather_deg=5.0, weight_smoothing=0)(image)
    smoothed: Image.Image = ClusterVersionRemap(correspondence, probability=1.0)(image)

    assert _totalVariation(image) < 12.0                       # the input is a flat surface
    assert _totalVariation(smoothed) < 0.5 * _totalVariation(speckled)


def testWeightSmoothingNeverLeaksIntoShadows() -> None:
    """Smoothing must not blur remap weight across the shadow floor: the hard protections are
    re-applied after the spatial smoothing."""
    lit_rgb: np.ndarray = np.asarray(_solidHsvImage(120.0, 0.6, 0.7))
    shadow_rgb: np.ndarray = np.asarray(_solidHsvImage(120.0, 0.6, 0.10))
    composite: np.ndarray = lit_rgb.copy()
    composite[:, : SIZE // 2] = shadow_rgb[:, : SIZE // 2]
    image: Image.Image = Image.fromarray(composite, mode="RGB")

    remap: ClusterVersionRemap = ClusterVersionRemap(
        {"green": _target("red", value_mean=0.7)}, probability=1.0, weight_smoothing=5)
    result: np.ndarray = np.asarray(remap(image))
    left_before: np.ndarray = composite[:, : SIZE // 2].astype(int)
    assert np.abs(result[:, : SIZE // 2].astype(int) - left_before).max() <= 3


def testRejectsUnsupportedSmoothingKernel() -> None:
    with pytest.raises(ValueError):
        ClusterVersionRemap({"blue": _target("green")}, weight_smoothing=4)
