"""Tests for turning a SAM label map into network input channels.

The property that matters most here is that nothing downstream depends on *which* id a
region was given. SAM numbers regions by decreasing area, an ordering that carries no colour
information, so any signal the encoder passes on must survive a relabelling. The area channel
must also be measured on the map as handed over -- after the crop -- or it describes a frame
the network is not being shown.

Naming/style follows the project convention (see CLAUDE.md).
"""
import numpy as np
import pytest
import torch

from pyaiwrap.segmentation_masks import (MaskEncoding, SegmentationMaskEncoder,
                                         createSegmentationMaskEncoder)


def _quadrantLabelMap(size: int = 8) -> np.ndarray:
    """A 2x2 partition: labels 1, 2, 3, 4, no background."""
    label_map: np.ndarray = np.zeros((size, size), dtype=np.uint8)
    half: int = size // 2
    label_map[:half, :half] = 1
    label_map[:half, half:] = 2
    label_map[half:, :half] = 3
    label_map[half:, half:] = 4
    return label_map


def _stripeLabelMap(size: int = 8) -> np.ndarray:
    """Left half region 1, right half background."""
    label_map: np.ndarray = np.zeros((size, size), dtype=np.uint8)
    label_map[:, :size // 2] = 1
    return label_map


def testChannelCountMatchesEncoding() -> None:
    """enc1's width is wired from this number, so it has to be right before anything trains."""
    assert SegmentationMaskEncoder(MaskEncoding.BOUNDARY_AREA).channels == 2
    assert SegmentationMaskEncoder(MaskEncoding.LABEL_ID).channels == 1
    assert SegmentationMaskEncoder(MaskEncoding.BOUNDARY_AREA_HASH, hash_channels=3).channels == 5
    assert SegmentationMaskEncoder(MaskEncoding.BOUNDARY_AREA_HASH, hash_channels=1).channels == 3


def testEncodedShapeAndRange() -> None:
    encoded: torch.Tensor = SegmentationMaskEncoder()(_quadrantLabelMap(8))

    assert encoded.shape == (2, 8, 8)
    assert encoded.dtype == torch.float32
    assert float(encoded.min()) >= 0.0 and float(encoded.max()) <= 1.0


def testBoundaryMarksBothSidesOfTheSeam() -> None:
    """A seam belongs to neither region more than the other, so both sides are marked."""
    boundary: torch.Tensor = SegmentationMaskEncoder()(_stripeLabelMap(8))[0]

    assert float(boundary[0, 3]) == 1.0        # last column of region 1
    assert float(boundary[0, 4]) == 1.0        # first column of background
    assert float(boundary[0, 2]) == 0.0        # interior stays clear
    assert float(boundary[0, 5]) == 0.0


def testBoundarySeparatesRegionFromBackground() -> None:
    """Background is not a region, but the edge against it is still an object outline."""
    boundary: torch.Tensor = SegmentationMaskEncoder()(_stripeLabelMap(8))[0]
    assert float(boundary[:, 3:5].min()) == 1.0


def testWiderBoundaryThickensTheContour() -> None:
    narrow: torch.Tensor = SegmentationMaskEncoder(boundary_width=1)(_stripeLabelMap(8))[0]
    wide: torch.Tensor = SegmentationMaskEncoder(boundary_width=3)(_stripeLabelMap(8))[0]

    assert float(wide.sum()) > float(narrow.sum())
    assert float(wide[0, 2]) == 1.0            # dilated into the region's interior


def testAreaIsSqrtOfTheRegionShare() -> None:
    area: torch.Tensor = SegmentationMaskEncoder()(_quadrantLabelMap(8))[1]

    expected: float = float(np.sqrt(0.25))
    assert float(area.min()) == pytest.approx(expected, abs=1e-6)
    assert float(area.max()) == pytest.approx(expected, abs=1e-6)


def testAreaIsZeroOnBackground() -> None:
    area: torch.Tensor = SegmentationMaskEncoder()(_stripeLabelMap(8))[1]

    assert float(area[0, 0]) == pytest.approx(float(np.sqrt(0.5)), abs=1e-6)
    assert float(area[0, 7]) == 0.0


def testAreaIsMeasuredOnTheMapAsGiven() -> None:
    """A crop that halves a region must halve the area it reports.

    The dataset hands the encoder the already-cropped map for exactly this reason: an area
    computed before the crop would describe pixels the network never sees.
    """
    encoder: SegmentationMaskEncoder = SegmentationMaskEncoder()
    full: np.ndarray = _stripeLabelMap(8)              # region 1 covers half the frame
    cropped: np.ndarray = full[:, :4]                  # ...and all of this crop

    assert float(encoder(full)[1].max()) == pytest.approx(float(np.sqrt(0.5)), abs=1e-6)
    assert float(encoder(cropped)[1].max()) == pytest.approx(1.0, abs=1e-6)


def testEncodingSurvivesRelabelling() -> None:
    """Relabelled regions, identical channels: the ids are names, not values."""
    encoder: SegmentationMaskEncoder = SegmentationMaskEncoder()
    label_map: np.ndarray = _quadrantLabelMap(8)
    relabelled: np.ndarray = np.array([0, 40, 7, 200, 13], dtype=np.uint8)[label_map]

    assert torch.equal(encoder(label_map), encoder(relabelled))


def testLabelIdEncodingIsTheRawMap() -> None:
    encoded: torch.Tensor = SegmentationMaskEncoder(MaskEncoding.LABEL_ID)(_quadrantLabelMap(8))

    assert encoded.shape == (1, 8, 8)
    assert float(encoded[0, 0, 0]) == pytest.approx(1.0 / 255.0, abs=1e-6)
    assert float(encoded[0, 7, 7]) == pytest.approx(4.0 / 255.0, abs=1e-6)


def testHashIsConstantInsideARegionAndDiffersBetweenRegions() -> None:
    encoded: torch.Tensor = SegmentationMaskEncoder(
        MaskEncoding.BOUNDARY_AREA_HASH, hash_channels=3
    )(_quadrantLabelMap(8))
    hash_channels: torch.Tensor = encoded[2:]

    top_left: torch.Tensor = hash_channels[:, :4, :4]
    assert torch.allclose(top_left, top_left[:, :1, :1].expand_as(top_left))

    top_right: torch.Tensor = hash_channels[:, :4, 4:]
    assert not torch.allclose(top_left[:, 0, 0], top_right[:, 0, 0])


def testHashIsZeroOnBackground() -> None:
    encoded: torch.Tensor = SegmentationMaskEncoder(
        MaskEncoding.BOUNDARY_AREA_HASH
    )(_stripeLabelMap(8))

    assert float(encoded[2:, :, 4:].abs().max()) == 0.0


def testRandomizedHashRedrawsPerItem() -> None:
    """The fingerprint the fixed table would give each image is what randomising removes."""
    encoder: SegmentationMaskEncoder = SegmentationMaskEncoder(
        MaskEncoding.BOUNDARY_AREA_HASH, randomize=True
    )
    label_map: np.ndarray = _quadrantLabelMap(8)

    assert not torch.equal(encoder(label_map)[2:], encoder(label_map)[2:])


def testFixedHashIsStableAcrossCallsAndInstances() -> None:
    first: SegmentationMaskEncoder = SegmentationMaskEncoder(
        MaskEncoding.BOUNDARY_AREA_HASH, randomize=False
    )
    second: SegmentationMaskEncoder = SegmentationMaskEncoder(
        MaskEncoding.BOUNDARY_AREA_HASH, randomize=False
    )
    label_map: np.ndarray = _quadrantLabelMap(8)

    assert torch.equal(first(label_map), first(label_map))
    assert torch.equal(first(label_map), second(label_map))


def testBoundaryAndAreaAreUnaffectedByHashRandomisation() -> None:
    """Randomising the hash must not leak into the deterministic channels."""
    encoder: SegmentationMaskEncoder = SegmentationMaskEncoder(
        MaskEncoding.BOUNDARY_AREA_HASH, randomize=True
    )
    label_map: np.ndarray = _quadrantLabelMap(8)

    assert torch.equal(encoder(label_map)[:2], encoder(label_map)[:2])


def testFactoryRejectsUnknownEncoding() -> None:
    with pytest.raises(ValueError, match="MASK_ENCODING"):
        createSegmentationMaskEncoder(encoding="segments")


def testFactoryBuildsTheConfiguredEncoder() -> None:
    encoder: SegmentationMaskEncoder = createSegmentationMaskEncoder(
        encoding="boundary_area_hash", boundary_width=2, hash_channels=4, randomize_hash=False
    )

    assert encoder.encoding is MaskEncoding.BOUNDARY_AREA_HASH
    assert encoder.channels == 6
    assert encoder.boundary_width == 2
    assert not encoder.randomize


def testRejectsMalformedInput() -> None:
    with pytest.raises(ValueError, match="2-D"):
        SegmentationMaskEncoder()(np.zeros((4, 4, 3), dtype=np.uint8))
