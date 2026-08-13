"""Tests for feeding SAM label maps through PairedImageFolder alongside the images.

Three things have to hold. The label map must be paired with the right image (by filename
stem, and loudly rather than silently when one is missing). It must go through the same
geometry as the image, which is only possible if the augmentation exposes its draw -- the
composed transform does not, and combining the two is rejected rather than quietly
misaligned. And the encoded channels must land behind the image's own, because that is where
the merge network splits them off for the trainable UNet.

Naming/style follows the project convention (see CLAUDE.md).
"""
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.segmentation_masks import MaskEncoding, SegmentationMaskEncoder
from pyaiwrap.transforms import createPairedGeometricAugmentation, createSharedGeometricAugmentation

IMAGE_SIZE: int = 32
MASK_SIZE: int = 64                 # deliberately unequal: SAM maps are 256 whatever the scan is
IMAGE_COUNT: int = 4


@pytest.fixture
def imageAndMaskFolders(tmp_path: Path) -> Tuple[Path, Path, List[str]]:
    """An ImageFolder and a flat mask folder describing the same left/right split.

    Left half of every image is black and belongs to region 1; the right half is white and
    belongs to region 2. Any geometry applied to one and not the other shows up as luminance
    disagreeing with the label under it.
    """
    image_root: Path = tmp_path / "imgs"
    class_dir: Path = image_root / "class0"
    class_dir.mkdir(parents=True)
    mask_root: Path = tmp_path / "masks"
    mask_root.mkdir()

    names: List[str] = []
    for index in range(IMAGE_COUNT):
        pixels: np.ndarray = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        pixels[:, IMAGE_SIZE // 2:, :] = 255
        Image.fromarray(pixels, mode="RGB").save(class_dir / f"img_{index}.png")

        label_map: np.ndarray = np.ones((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
        label_map[:, MASK_SIZE // 2:] = 2
        Image.fromarray(label_map, mode="L").save(mask_root / f"img_{index}.png")
        names.append(f"img_{index}.png")

    return image_root, mask_root, names


def _grayscaleTransform() -> transforms.Compose:
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor()])


def _toTensorTransform() -> transforms.Compose:
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                               transforms.ToTensor()])


def _buildDataset(image_root: Path, mask_root: Path, augmentation=None,
                  encoder: SegmentationMaskEncoder = None) -> PairedImageFolder:
    return PairedImageFolder(
        str(image_root), _grayscaleTransform(), _toTensorTransform(),
        shared_augmentation=augmentation,
        mask_folder_path=str(mask_root),
        mask_encoder=encoder or SegmentationMaskEncoder(),
        image_size=IMAGE_SIZE
    )


def testMaskChannelsAreAppendedToTheInput(imageAndMaskFolders) -> None:
    image_root, mask_root, _ = imageAndMaskFolders
    model_input, target, _, _ = _buildDataset(image_root, mask_root)[0]

    assert model_input.shape == (3, IMAGE_SIZE, IMAGE_SIZE)   # luminance + boundary + area
    assert target.shape == (3, IMAGE_SIZE, IMAGE_SIZE)        # the target is untouched


def testImageChannelsComeFirst(imageAndMaskFolders) -> None:
    """The merge network splits at the extractors' channel count, so order is load-bearing."""
    image_root, mask_root, _ = imageAndMaskFolders
    dataset: PairedImageFolder = _buildDataset(image_root, mask_root)
    model_input, _, _, _ = dataset[0]

    raw: Image.Image = Image.open(image_root / "class0" / "img_0.png").convert("RGB")
    assert torch.allclose(model_input[:1], _grayscaleTransform()(raw))


def testChannelCountFollowsTheEncoding(imageAndMaskFolders) -> None:
    image_root, mask_root, _ = imageAndMaskFolders
    encoder: SegmentationMaskEncoder = SegmentationMaskEncoder(
        MaskEncoding.BOUNDARY_AREA_HASH, hash_channels=3
    )
    model_input, _, _, _ = _buildDataset(image_root, mask_root, encoder=encoder)[0]

    assert model_input.shape[0] == 1 + encoder.channels


def testMaskIsSquaredOffToTheInputSize(imageAndMaskFolders) -> None:
    """Validation has no augmentation to resample the 64x64 map down to the 32x32 input."""
    image_root, mask_root, _ = imageAndMaskFolders
    model_input, _, _, _ = _buildDataset(image_root, mask_root)[0]

    assert model_input.shape[-2:] == (IMAGE_SIZE, IMAGE_SIZE)


def testMaskAlignsWithTheImageWithoutAugmentation(imageAndMaskFolders) -> None:
    image_root, mask_root, _ = imageAndMaskFolders
    model_input, _, _, _ = _buildDataset(image_root, mask_root)[0]

    boundary: torch.Tensor = model_input[1]
    assert float(boundary[:, IMAGE_SIZE // 2 - 1:IMAGE_SIZE // 2 + 1].min()) == 1.0
    assert float(boundary[:, :IMAGE_SIZE // 2 - 1].max()) == 0.0


def _luminanceEdgeColumn(luminance: np.ndarray) -> int:
    """Column where the black half meets the white one, or -1 if the row is uniform."""
    steps: np.ndarray = np.abs(np.diff(luminance[0]))
    return int(np.argmax(steps)) if steps.max(initial=0.0) > 0.3 else -1


def _boundaryColumn(boundary: np.ndarray) -> int:
    """Column the mask says the seam is in, or -1 if it marks none."""
    marked: np.ndarray = np.flatnonzero(boundary[0] > 0.5)
    return int(marked[0]) if marked.size else -1


def testMaskFollowsTheImageThroughAugmentation(imageAndMaskFolders) -> None:
    """The seam in the mask must stay on the seam in the luminance, crop after crop.

    A crop that lands inside one region has no seam in either, which is agreement too; what
    must never happen is one of them claiming an edge the other does not have.
    """
    image_root, mask_root, _ = imageAndMaskFolders
    dataset: PairedImageFolder = _buildDataset(
        image_root, mask_root,
        augmentation=createPairedGeometricAugmentation(IMAGE_SIZE, flip_probability=0.5,
                                                       crop_scale_min=0.4)
    )

    saw_seam: bool = False
    for _ in range(40):
        model_input, _, _, _ = dataset[0]
        image_edge: int = _luminanceEdgeColumn(model_input[0].numpy())
        mask_edge: int = _boundaryColumn(model_input[1].numpy())

        assert (image_edge == -1) == (mask_edge == -1), "only one of the two sees a seam"
        if image_edge != -1:
            saw_seam = True
            # One pixel of slack: the image is resampled bilinearly and the mask by nearest,
            # so the two round the seam's sub-pixel position independently.
            assert abs(image_edge - mask_edge) <= 1
    assert saw_seam, "no crop straddled the seam"


def testAreaIsRecomputedForTheCrop(imageAndMaskFolders) -> None:
    """A crop that lands inside one region must report that region as filling the frame."""
    image_root, mask_root, _ = imageAndMaskFolders
    dataset: PairedImageFolder = _buildDataset(
        image_root, mask_root,
        augmentation=createPairedGeometricAugmentation(IMAGE_SIZE, flip_probability=0.0,
                                                       crop_scale_min=0.1, crop_scale_max=0.2)
    )

    saw_single_region_crop: bool = False
    for _ in range(60):
        model_input, _, _, _ = dataset[0]
        if float(model_input[1].max()) == 0.0:            # no boundary: one region only
            saw_single_region_crop = True
            assert float(model_input[2].min()) == pytest.approx(1.0, abs=1e-5)
    assert saw_single_region_crop, "no crop landed inside a single region"


def testEveryItemIsPairedWithItsOwnMask(imageAndMaskFolders) -> None:
    """Pairing is by stem, not by directory order, which ImageFolder does not guarantee."""
    image_root, mask_root, names = imageAndMaskFolders
    marked: np.ndarray = np.full((MASK_SIZE, MASK_SIZE), 3, dtype=np.uint8)
    Image.fromarray(marked, mode="L").save(mask_root / names[2])

    dataset: PairedImageFolder = _buildDataset(image_root, mask_root)

    assert float(dataset[2][0][1].max()) == 0.0           # the marked map has no seam
    assert float(dataset[0][0][1].max()) == 1.0


def testMissingMaskIsRejected(imageAndMaskFolders) -> None:
    image_root, mask_root, names = imageAndMaskFolders
    (mask_root / names[1]).unlink()

    with pytest.raises(ValueError, match="No label map"):
        _buildDataset(image_root, mask_root)


def testComposedAugmentationIsRejected(imageAndMaskFolders) -> None:
    """It samples its crop inside __call__, so the mask cannot be given the same geometry."""
    image_root, mask_root, _ = imageAndMaskFolders

    with pytest.raises(ValueError, match="PairedGeometricAugmentation"):
        _buildDataset(image_root, mask_root,
                      augmentation=createSharedGeometricAugmentation(IMAGE_SIZE))


def testEdgesAndMasksCannotBeCombined(tmp_path: Path, imageAndMaskFolders) -> None:
    image_root, mask_root, names = imageAndMaskFolders
    pairing: Path = tmp_path / "pairing.csv"
    with open(pairing, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_filepath", "edges_filepath"])
        for name in names:
            writer.writerow([name, str(mask_root / name)])

    with pytest.raises(ValueError, match="pick one"):
        PairedImageFolder(str(image_root), _grayscaleTransform(), _toTensorTransform(),
                          segmentation_pairing=str(pairing), mask_folder_path=str(mask_root),
                          image_size=IMAGE_SIZE)


def testImageSizeIsInferredFromTheInputTransform(imageAndMaskFolders) -> None:
    """Config always passes it, but the dataset must not depend on the caller remembering."""
    image_root, mask_root, _ = imageAndMaskFolders
    dataset: PairedImageFolder = PairedImageFolder(
        str(image_root), _grayscaleTransform(), _toTensorTransform(),
        mask_folder_path=str(mask_root)
    )

    assert dataset[0][0].shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def testUnconditionedDatasetIsUnchanged(imageAndMaskFolders) -> None:
    """The whole point of keeping masks opt-in: old configs must produce the old tensors."""
    image_root, _, _ = imageAndMaskFolders
    dataset: PairedImageFolder = PairedImageFolder(
        str(image_root), _grayscaleTransform(), _toTensorTransform()
    )
    model_input, _, _, _ = dataset[0]

    raw: Image.Image = Image.open(image_root / "class0" / "img_0.png").convert("RGB")
    assert model_input.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.allclose(model_input, _grayscaleTransform()(raw))
