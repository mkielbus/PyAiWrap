"""Train-time wiring for the L5b cluster-version remap (Phase 1b).

Bridges the pure planner (`remap_planner`) to the pixel transform (`transforms`): for each
training sample it looks up that image's cluster and colour version, asks the planner for a
correspondence, and applies a freshly-built `ClusterVersionRemap`. A new correspondence is drawn
every epoch, so one image sees several plausible recolours over training rather than a single
fixed one -- that variety is the point of the augmentation.

Layering: transforms.py holds the pixel op, remap_planner.py the pure decision logic, and this
module the I/O-ish glue (metadata lookup, per-worker RNG). Nothing here is imported by the other
two, so the planner stays unit-testable without image or dataset machinery.
"""

import csv
import os
import random
from typing import Callable, Dict, FrozenSet, Optional, Tuple

from PIL import Image

from pyaiwrap.remap_planner import (ClusterProfile, ClusterVersion, RemapPlanner,
                                    RemapTarget, loadClusterProfiles)
from pyaiwrap.transforms import ClusterVersionRemap, PathAwareImageTransform

# filename -> (cluster id, the colour version of that particular image)
ImageMetadata = Dict[str, Tuple[int, ClusterVersion]]


def _splitColorSet(pipe_joined: str) -> FrozenSet[str]:
    return frozenset(part for part in pipe_joined.split("|") if part)


def loadImageMetadata(image_versions_path: str, split: Optional[str] = None) -> ImageMetadata:
    """Read the Phase 0.4 per-image sidecar (image_versions.csv) into a filename lookup.

    `split` filters on the split_v2 column (pass "train" to keep only training images, so a
    stray val/test file can never be augmented); None keeps every row.
    """
    metadata: ImageMetadata = {}
    with open(image_versions_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["split_v2"] != split:
                continue
            achromatic: FrozenSet[str] = _splitColorSet(row["achromatic_set"])
            chromatic: FrozenSet[str] = _splitColorSet(row["version"]) - achromatic
            metadata[row["filename"]] = (
                int(row["cluster_k98"]),
                ClusterVersion(chromatic=chromatic, achromatic=achromatic, count=1))
    return metadata


class ClusterVersionRemapAugmentation(PathAwareImageTransform):
    """Target-side augmentation: recolour an image to another colour version of its own cluster.

    Images with no metadata entry (not in the inventory, or filtered out by split) and clusters
    the planner refuses (blacklisted, fully frozen, no eligible target version) are passed
    through unchanged, so enabling this can never corrupt a sample it does not understand.
    """

    def __init__(self, planner: RemapPlanner, profiles: Dict[int, ClusterProfile],
                 metadata: ImageMetadata, probability: float = 0.3,
                 seed: int = 0, rng: Optional[random.Random] = None) -> None:
        self.planner: RemapPlanner = planner
        self.profiles: Dict[int, ClusterProfile] = profiles
        self.metadata: ImageMetadata = metadata
        self.probability: float = probability
        self.seed: int = seed
        self._rng: random.Random = rng if rng is not None else random.Random(seed)
        self._rng_pid: int = os.getpid()

    def _ensureWorkerRng(self) -> None:
        """Re-seed per worker process. DataLoader workers are forked, so they would otherwise
        inherit one RNG state and draw the SAME sequence of target versions -- collapsing the
        augmentation's variety exactly where it is supposed to come from."""
        pid: int = os.getpid()
        if pid == self._rng_pid:
            return
        self._rng = random.Random(f"{self.seed}-{pid}")   # str seed: tuples are rejected in 3.14
        self.planner.reseed(self._rng.getrandbits(63))
        self._rng_pid = pid

    def __call__(self, img: Image.Image, image_path: str) -> Image.Image:
        self._ensureWorkerRng()
        if self._rng.random() >= self.probability:
            return img
        entry: Optional[Tuple[int, ClusterVersion]] = self.metadata.get(
            os.path.basename(image_path))
        if entry is None:
            return img                                   # unknown image: never guess a remap
        cluster_id, version = entry
        profile: Optional[ClusterProfile] = self.profiles.get(cluster_id)
        if profile is None:
            return img
        correspondence: Dict[str, RemapTarget] = self.planner.planCorrespondence(profile, version)
        if not correspondence:
            return img                                   # frozen / blacklisted / no target
        return ClusterVersionRemap(correspondence, probability=1.0)(img)


def createClusterVersionRemapAugmentation(
        version_inventory_path: str, color_sv_path: str, image_versions_path: str,
        cluster_names_path: Optional[str] = None, probability: float = 0.3,
        freeze_threshold: Optional[float] = None, min_support: Optional[int] = None,
        split: Optional[str] = "train", seed: int = 0) -> ClusterVersionRemapAugmentation:
    """Build the augmentation from the Phase 0 analysis artifacts.

    freeze_threshold/min_support default to the planner's own constants when None, so the
    reviewed QA settings are not silently re-specified in two places.
    """
    profiles: Dict[int, ClusterProfile] = loadClusterProfiles(
        version_inventory_path, color_sv_path, cluster_names_path=cluster_names_path)
    metadata: ImageMetadata = loadImageMetadata(image_versions_path, split=split)
    planner_kwargs: Dict[str, object] = {"rng": random.Random(seed)}
    if freeze_threshold is not None:
        planner_kwargs["freeze_threshold"] = freeze_threshold
    if min_support is not None:
        planner_kwargs["min_support"] = min_support
    planner: RemapPlanner = RemapPlanner(**planner_kwargs)
    return ClusterVersionRemapAugmentation(planner, profiles, metadata,
                                           probability=probability, seed=seed)
