"""Tests for the train-time wiring of the L5b cluster-version remap.

Covers the metadata lookup, the pass-through guarantees (unknown image, unknown cluster,
blacklisted/frozen cluster), the probability gate, and the per-worker RNG de-correlation.
"""

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
from PIL import Image

from pyaiwrap.remap_augmentation import (ClusterVersionRemapAugmentation,
                                         createClusterVersionRemapAugmentation, loadImageMetadata)
from pyaiwrap.remap_planner import (ColorStats, ClusterProfile, ClusterVersion, RemapPlanner)
from pyaiwrap.transforms import ComposedTargetAugmentation, PathAwareImageTransform

SIZE: int = 48


def makeStats(saturation: float = 0.5, value: float = 0.6) -> ColorStats:
    return ColorStats(saturation_mean=saturation, saturation_std=0.1,
                      value_mean=value, value_std=0.1)


def fieldProfile(cluster_id: int = 1) -> ClusterProfile:
    """Blue is invariant (frozen); the ground colour varies green/yellow/brown."""
    achromatic: frozenset = frozenset({"gray"})
    versions: tuple = (
        ClusterVersion(chromatic=frozenset({"blue", "green"}), achromatic=achromatic, count=120),
        ClusterVersion(chromatic=frozenset({"blue", "yellow"}), achromatic=achromatic, count=80),
        ClusterVersion(chromatic=frozenset({"blue", "brown"}), achromatic=achromatic, count=40),
    )
    stats: dict = {c: makeStats() for c in ("blue", "green", "yellow", "brown")}
    return ClusterProfile(cluster_id=cluster_id, versions=versions, color_stats=stats)


def greenImage() -> Image.Image:
    hsv: np.ndarray = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    hsv[..., 0] = int(round(120.0 / 2.0))
    hsv[..., 1] = int(round(0.6 * 255))
    hsv[..., 2] = int(round(0.7 * 255))
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mode="RGB")


def buildAugmentation(probability: float = 1.0, cluster_id: int = 1,
                      blacklist: frozenset = frozenset()) -> ClusterVersionRemapAugmentation:
    profile: ClusterProfile = fieldProfile(cluster_id)
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.90, min_support=10,
                                         blacklist=blacklist, rng=random.Random(0))
    metadata: dict = {"field.jpg": (cluster_id, profile.versions[0])}
    return ClusterVersionRemapAugmentation(planner, {cluster_id: profile}, metadata,
                                           probability=probability, seed=0)


def changed(before: Image.Image, after: Image.Image) -> bool:
    return not np.array_equal(np.asarray(before), np.asarray(after))


def testAugmentationIsPathAware() -> None:
    assert isinstance(buildAugmentation(), PathAwareImageTransform)


def testRemapsAnImageItHasMetadataFor() -> None:
    augmentation: ClusterVersionRemapAugmentation = buildAugmentation(probability=1.0)
    image: Image.Image = greenImage()
    assert changed(image, augmentation(image, "/data/train/field.jpg"))


def testUnknownFilenameIsPassedThrough() -> None:
    augmentation: ClusterVersionRemapAugmentation = buildAugmentation(probability=1.0)
    image: Image.Image = greenImage()
    assert not changed(image, augmentation(image, "/data/train/not_in_inventory.jpg"))


def testUnknownClusterIsPassedThrough() -> None:
    augmentation: ClusterVersionRemapAugmentation = buildAugmentation(probability=1.0)
    augmentation.profiles = {}                       # metadata points at a cluster we lack
    image: Image.Image = greenImage()
    assert not changed(image, augmentation(image, "/data/train/field.jpg"))


def testBlacklistedClusterIsPassedThrough() -> None:
    augmentation: ClusterVersionRemapAugmentation = buildAugmentation(
        probability=1.0, blacklist=frozenset({1}))
    image: Image.Image = greenImage()
    assert not changed(image, augmentation(image, "/data/train/field.jpg"))


def testProbabilityZeroIsIdentity() -> None:
    augmentation: ClusterVersionRemapAugmentation = buildAugmentation(probability=0.0)
    image: Image.Image = greenImage()
    for _ in range(10):
        assert not changed(image, augmentation(image, "/data/train/field.jpg"))


def testProbabilityGateFiresRoughlyAtTheGivenRate() -> None:
    augmentation: ClusterVersionRemapAugmentation = buildAugmentation(probability=0.3)
    image: Image.Image = greenImage()
    hits: int = sum(changed(image, augmentation(image, "/data/train/field.jpg"))
                    for _ in range(400))
    assert 0.2 < hits / 400 < 0.4


def testRepeatedCallsDrawDifferentTargets() -> None:
    """The same image must see several plausible recolours over training, not one fixed one."""
    augmentation: ClusterVersionRemapAugmentation = buildAugmentation(probability=1.0)
    image: Image.Image = greenImage()
    outputs: set = {np.asarray(augmentation(image, "/data/train/field.jpg")).tobytes()
                    for _ in range(40)}
    assert len(outputs) > 1


def testWorkerRngIsReseededPerProcess() -> None:
    """Forked workers must not inherit one RNG state and draw identical target sequences."""
    augmentation: ClusterVersionRemapAugmentation = buildAugmentation(probability=1.0)
    augmentation._rng_pid = os.getpid() - 1          # simulate having been forked
    first: List[float] = [augmentation._rng.random() for _ in range(3)]

    other: ClusterVersionRemapAugmentation = buildAugmentation(probability=1.0)
    other._rng_pid = os.getpid() - 2
    other._ensureWorkerRng()
    assert other._rng_pid == os.getpid()
    assert [other._rng.random() for _ in range(3)] != first


def testComposedTargetAugmentationPassesPathOnlyWhereNeeded() -> None:
    seen: dict = {}

    def plain(image: Image.Image) -> Image.Image:
        seen["plain"] = True
        return image

    class Recorder(PathAwareImageTransform):
        def __call__(self, image: Image.Image, image_path: str) -> Image.Image:
            seen["path"] = image_path
            return image

    composed: ComposedTargetAugmentation = ComposedTargetAugmentation([plain, Recorder()])
    image: Image.Image = greenImage()
    composed(image, "/data/train/field.jpg")
    assert seen["plain"] is True
    assert seen["path"] == "/data/train/field.jpg"


def writeCsv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def testLoadImageMetadataFiltersBySplit(tmp_path: Path) -> None:
    path: Path = tmp_path / "image_versions.csv"
    writeCsv(path, ["filename", "split_v2", "cluster_k98", "version", "achromatic_set"],
             [["a.jpg", "train", "3", "blue|green|gray", "gray"],
              ["b.jpg", "val", "3", "blue|brown|gray", "gray"]])
    train_only: dict = loadImageMetadata(str(path), split="train")
    assert set(train_only) == {"a.jpg"}
    cluster_id, version = train_only["a.jpg"]
    assert cluster_id == 3
    assert version.chromatic == frozenset({"blue", "green"})
    assert version.achromatic == frozenset({"gray"})
    assert set(loadImageMetadata(str(path), split=None)) == {"a.jpg", "b.jpg"}


def testFactoryBuildsFromCsvArtifacts(tmp_path: Path) -> None:
    inventory: Path = tmp_path / "version_inventory.csv"
    writeCsv(inventory, ["cluster_k98", "version", "achromatic_set", "n_train"],
             [["1", "blue|green|gray", "gray", "120"],
              ["1", "blue|yellow|gray", "gray", "80"]])
    color_sv: Path = tmp_path / "cluster_color_sv.csv"
    writeCsv(color_sv, ["cluster_k98", "color", "saturation_mean", "saturation_std",
                        "value_mean", "value_std"],
             [["1", color, "0.5", "0.1", "0.6", "0.1"] for color in ("blue", "green", "yellow")])
    versions: Path = tmp_path / "image_versions.csv"
    writeCsv(versions, ["filename", "split_v2", "cluster_k98", "version", "achromatic_set"],
             [["field.jpg", "train", "1", "blue|green|gray", "gray"]])

    augmentation: ClusterVersionRemapAugmentation = createClusterVersionRemapAugmentation(
        str(inventory), str(color_sv), str(versions), probability=1.0, freeze_threshold=0.90,
        min_support=10, seed=0)
    image: Image.Image = greenImage()
    assert changed(image, augmentation(image, "/data/train/field.jpg"))
    assert not changed(image, augmentation(image, "/data/train/missing.jpg"))
