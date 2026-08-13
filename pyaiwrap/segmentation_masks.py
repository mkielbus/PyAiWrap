"""Turning a SAM label map into channels a colorization network can actually use.

The exporter (analysis/remote_masks/sam_export_masks.py) writes one uint8 PNG per image:
a hard partition of the 256x256 frame, 0 for the pixels no mask claimed, then 1..N for the
regions in decreasing area order. Those ids are *names*. Feeding them to a convolution as a
number would ask the network to read meaning into an ordering that has none beyond "bigger
first", and to treat the step between region 7 and region 8 as a quantity.

So the map is re-expressed as channels that do not depend on which name a region was given:

  * boundary -- 1 where a pixel's 4-neighbourhood contains a different label. This is where
    colour is allowed to change abruptly, at full input resolution and already in enc1. The
    frozen ConvNeXt branch cannot supply it: it is injected at the bottleneck, at stride 8,
    so an object edge inside an 8x8 cell is not recoverable from it.
  * area -- sqrt of the region's share of the visible frame, constant inside a region, 0 on
    background. Object scale is a colour prior (a region covering half the frame is sky or
    wall or backdrop far more often than it is an object). The square root is there because
    the region-size distribution is heavily skewed: without it most regions sit below 0.02
    and the channel is a flat line.
  * hash -- optional, `hash_channels` values per region, constant inside it and otherwise
    meaningless. The only thing learnable from a meaningless value is the equality relation:
    "these pixels carry the same value, so they are one object and should share a colour",
    which is the one signal boundary and area cannot give (two adjacent regions of similar
    size are separated by nothing but a contour line). Off by default -- see `randomize` for
    the reason it is drawn afresh per item when it is on.

Naming/style follows the project convention (see CLAUDE.md).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

# The exporter's label map is uint8 with 0 reserved for background, so ids never exceed this.
MAX_LABELS: int = 255
LABEL_VALUES: int = MAX_LABELS + 1

# Fixed seed for the deterministic hash table: the same region id then encodes to the same
# vector in every process, which is what makes a checkpoint's behaviour reproducible.
HASH_SEED: int = 0


class MaskEncoding(Enum):
    """How a label map becomes network input channels."""

    BOUNDARY_AREA = "boundary_area"
    BOUNDARY_AREA_HASH = "boundary_area_hash"
    LABEL_ID = "label_id"


class SegmentationMaskEncoder:
    """Encode a uint8 label map as [C, H, W] float32 channels in [0, 1].

    randomize (hash encodings only): draw the per-region vectors afresh for every item
    instead of deriving them from the region id. A table derived from the id is constant for
    a given image across the whole run, which makes it a fingerprint of that image -- extra
    capacity for memorising the training set, and this model's measured train/val gap is
    already ~60%. Re-drawing keeps exactly the property the channel is for (equal inside a
    region, different between regions) and destroys the one it should not have (the same
    image always presenting the same pattern). Validation draws too, so the two match.
    """

    def __init__(self, encoding: MaskEncoding = MaskEncoding.BOUNDARY_AREA,
                 boundary_width: int = 1, hash_channels: int = 3,
                 randomize: bool = True) -> None:
        if boundary_width < 1:
            raise ValueError(f"boundary_width must be at least 1, got {boundary_width}")
        if hash_channels < 1:
            raise ValueError(f"hash_channels must be at least 1, got {hash_channels}")

        self.encoding: MaskEncoding = encoding
        self.boundary_width: int = boundary_width
        self.hash_channels: int = hash_channels
        self.randomize: bool = randomize

        self._fixed_hash_table: Optional[np.ndarray] = None
        if encoding is MaskEncoding.BOUNDARY_AREA_HASH and not randomize:
            self._fixed_hash_table = self._buildHashTable(np.random.default_rng(HASH_SEED))

    @property
    def channels(self) -> int:
        """How many channels this encoder appends, i.e. how much wider enc1 has to be."""
        if self.encoding is MaskEncoding.LABEL_ID:
            return 1
        if self.encoding is MaskEncoding.BOUNDARY_AREA:
            return 2
        return 2 + self.hash_channels

    def __call__(self, label_map: np.ndarray) -> torch.Tensor:
        if label_map.ndim != 2:
            raise ValueError(f"label map must be 2-D (H, W), got shape {label_map.shape}")
        labels: np.ndarray = label_map.astype(np.int64, copy=False)
        if labels.max(initial=0) > MAX_LABELS:
            raise ValueError(f"label ids must fit in uint8, got max {labels.max()}")

        if self.encoding is MaskEncoding.LABEL_ID:
            channels = [labels.astype(np.float32) / MAX_LABELS]
        else:
            channels = [self._boundaryChannel(labels), self._areaChannel(labels)]
            if self.encoding is MaskEncoding.BOUNDARY_AREA_HASH:
                channels.extend(self._hashChannels(labels))

        return torch.from_numpy(np.stack(channels).astype(np.float32, copy=False))

    def _boundaryChannel(self, labels: np.ndarray) -> np.ndarray:
        """1 where a 4-neighbour carries a different label, marked on both sides of the seam.

        Both sides matter: a one-sided difference map would sit inside whichever region
        happens to come first in memory order, which is an arbitrary choice for a symmetric
        relation, and would leave the other region's own outline incomplete.
        """
        boundary: np.ndarray = np.zeros(labels.shape, dtype=bool)

        horizontal: np.ndarray = labels[:, :-1] != labels[:, 1:]
        boundary[:, :-1] |= horizontal
        boundary[:, 1:] |= horizontal

        vertical: np.ndarray = labels[:-1, :] != labels[1:, :]
        boundary[:-1, :] |= vertical
        boundary[1:, :] |= vertical

        if self.boundary_width > 1:
            # A 1 px contour survives enc1 but is thinned away by the stride-2 downsamples;
            # widening it trades exact localisation for reaching the deeper encoder stages.
            boundary = binary_dilation(boundary, iterations=self.boundary_width - 1)

        return boundary.astype(np.float32)

    @staticmethod
    def _areaChannel(labels: np.ndarray) -> np.ndarray:
        """sqrt of each region's share of the frame, 0 on background.

        Counted on the label map as it is handed over -- after the crop, never before. A
        region half of which the crop threw away must read as the size it now occupies, or
        the channel describes a frame the network is not being shown.
        """
        counts: np.ndarray = np.bincount(labels.reshape(-1), minlength=LABEL_VALUES)
        fractions: np.ndarray = np.sqrt(counts.astype(np.float32) / labels.size)
        fractions[0] = 0.0                      # background is not a region
        return fractions[labels]

    def _hashChannels(self, labels: np.ndarray) -> list:
        table: np.ndarray = (self._fixed_hash_table if self._fixed_hash_table is not None
                             else self._buildHashTable(np.random.default_rng()))
        painted: np.ndarray = table[labels]     # [H, W, hash_channels]
        return [painted[:, :, index] for index in range(self.hash_channels)]

    def _buildHashTable(self, generator: np.random.Generator) -> np.ndarray:
        table: np.ndarray = generator.random((LABEL_VALUES, self.hash_channels), dtype=np.float32)
        table[0] = 0.0                          # background reads as "no region", not as a region
        return table


def createSegmentationMaskEncoder(encoding: str = MaskEncoding.BOUNDARY_AREA.value,
                                  boundary_width: int = 1, hash_channels: int = 3,
                                  randomize_hash: bool = True) -> SegmentationMaskEncoder:
    """Build the encoder from config values (MASK_ENCODING and friends)."""
    try:
        encoding_enum: MaskEncoding = MaskEncoding(encoding)
    except ValueError:
        raise ValueError(
            f"MASK_ENCODING must be one of {[member.value for member in MaskEncoding]}, "
            f"got {encoding!r}"
        )
    return SegmentationMaskEncoder(
        encoding=encoding_enum,
        boundary_width=boundary_width,
        hash_channels=hash_channels,
        randomize=randomize_hash
    )


def loadLabelMap(path: str) -> Image.Image:
    """Read an exported label map as a single-channel PIL image.

    PIL is used rather than cv2 so the map stays a PIL image all the way through the
    geometric augmentation, which is what the image branch works in too.
    """
    label_map: Image.Image = Image.open(path)
    if label_map.mode != "L":
        label_map = label_map.convert("L")
    return label_map
