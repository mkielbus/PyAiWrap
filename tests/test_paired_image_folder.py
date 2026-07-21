"""Tests for PairedImageFolder, covering the shared paired-augmentation rework.

Split into two groups:
  * safety-net tests characterise the behaviour that must NOT change when
    augmentation is off (the pre-refactor contract);
  * new-behaviour tests pin the shared-augmentation invariants introduced by
    Phase 1 of the anti-overfitting plan (one geometric op sampled once per
    item and applied identically to input and target; val stays deterministic).

The tests build a tiny synthetic ImageFolder in a temp dir so they never depend
on the real dataset.

Naming/style follows the project convention (see CLAUDE.md): functionName for
functions/methods, variable_name for variables, _privateHelper for module-private
helpers, and `:` type specifiers on args/returns/variables. pytest still discovers
the testXxx callables via its test* glob.
"""
import csv
import os
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
from PIL import Image
from torchvision import transforms

from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.transforms import PathAwareImageTransform

IMAGE_SIZE: int = 32


def _writeImage(path: Path, pixels: torch.Tensor) -> None:
    """pixels: (H, W, 3) uint8 tensor -> saved RGB PNG."""
    image: Image.Image = Image.fromarray(pixels.to(torch.uint8).numpy(), mode="RGB")
    image.save(path)


@pytest.fixture
def imageFolder(tmp_path: Path) -> Tuple[Path, List[str]]:
    """A one-class ImageFolder with horizontally asymmetric images.

    Left half black, right half white, so a horizontal flip is detectable and
    a crop is spatially meaningful. Returns (root, [filenames]).
    """
    root: Path = tmp_path / "imgs"
    class_dir: Path = root / "class0"
    class_dir.mkdir(parents=True)
    names: List[str] = []
    for i in range(4):
        pixels: torch.Tensor = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.uint8)
        pixels[:, IMAGE_SIZE // 2:, :] = 255       # right half white
        pixels[0, 0, :] = i * 10                   # per-image marker pixel
        name: str = f"img_{i}.png"
        _writeImage(class_dir / name, pixels)
        names.append(name)
    return root, names


def _toTensorTransform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([transforms.Resize((image_size, image_size)),
                               transforms.ToTensor()])


def _grayscaleTransform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([transforms.Resize((image_size, image_size)),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor()])


# --------------------------------------------------------------------------
# safety-net: behaviour with augmentation OFF must match the old contract
# --------------------------------------------------------------------------

def testLengthAndPairingWithoutAug(imageFolder: Tuple[Path, List[str]]) -> None:
    root, names = imageFolder
    dataset: PairedImageFolder = PairedImageFolder(
        str(root), _grayscaleTransform(), _toTensorTransform())
    assert len(dataset) == len(names)
    model_input, target, _, _ = dataset[0]
    assert model_input.shape == (1, IMAGE_SIZE, IMAGE_SIZE)   # grayscale input
    assert target.shape == (3, IMAGE_SIZE, IMAGE_SIZE)        # rgb target


def testOutputMatchesTransformsAppliedToSameImage(imageFolder: Tuple[Path, List[str]]) -> None:
    """No-aug output must equal running each transform on the loaded PIL image."""
    root, _ = imageFolder
    input_transform: transforms.Compose = _grayscaleTransform()
    target_transform: transforms.Compose = _toTensorTransform()
    dataset: PairedImageFolder = PairedImageFolder(str(root), input_transform, target_transform)

    raw: Image.Image = Image.open(root / "class0" / "img_0.png").convert("RGB")
    model_input, target, _, _ = dataset[0]
    assert torch.allclose(model_input, input_transform(raw))
    assert torch.allclose(target, target_transform(raw))


def testValStyleAccessIsDeterministic(imageFolder: Tuple[Path, List[str]]) -> None:
    """Without augmentation, repeated access to the same index is identical."""
    root, _ = imageFolder
    dataset: PairedImageFolder = PairedImageFolder(
        str(root), _toTensorTransform(), _toTensorTransform())
    first_input, first_target, _, _ = dataset[1]
    second_input, second_target, _, _ = dataset[1]
    assert torch.equal(first_input, second_input)
    assert torch.equal(first_target, second_target)


# --------------------------------------------------------------------------
# new behaviour: shared geometric augmentation
# --------------------------------------------------------------------------

def testSharedFlipAppliedIdenticallyToInputAndTarget(imageFolder: Tuple[Path, List[str]]) -> None:
    """A deterministic horizontal flip must show up in both input and target."""
    root, _ = imageFolder
    flip: transforms.RandomHorizontalFlip = transforms.RandomHorizontalFlip(p=1.0)  # always flips
    dataset: PairedImageFolder = PairedImageFolder(
        str(root), _toTensorTransform(), _toTensorTransform(), shared_augmentation=flip)

    raw: Image.Image = Image.open(root / "class0" / "img_0.png").convert("RGB")
    flipped: torch.Tensor = _toTensorTransform()(raw.transpose(Image.FLIP_LEFT_RIGHT))

    model_input, target, _, _ = dataset[0]
    assert torch.allclose(model_input, target)     # same geometric op on both
    assert torch.allclose(model_input, flipped)    # and it is the flip we asked for


def testRandomCropSampledOncePerItem(imageFolder: Tuple[Path, List[str]]) -> None:
    """The random geometric op is sampled ONCE, so input and target align.

    With the old two-ImageFolder design a random crop would be resampled
    independently for input and target; here they must be pixel-identical.
    """
    root, _ = imageFolder
    crop: transforms.RandomResizedCrop = transforms.RandomResizedCrop(
        IMAGE_SIZE, scale=(0.4, 0.6), ratio=(0.9, 1.1))
    dataset: PairedImageFolder = PairedImageFolder(
        str(root), _toTensorTransform(), _toTensorTransform(), shared_augmentation=crop)
    for _ in range(5):  # several draws to defeat a lucky match
        model_input, target, _, _ = dataset[2]
        assert torch.allclose(model_input, target)


def testSharedAugmentationAndEdgesAreMutuallyExclusive(
        imageFolder: Tuple[Path, List[str]], tmp_path: Path) -> None:
    """Combining a geometric aug with the (un-augmented) edges path would
    silently misalign edges, so it must be rejected explicitly."""
    root, names = imageFolder
    pairing: Path = tmp_path / "pairing.csv"
    with open(pairing, "w", newline="") as pairing_file:
        writer = csv.writer(pairing_file)
        writer.writerow(["image_filepath", "edges_filepath"])
        for name in names:
            image_path: str = str(root / "class0" / name)
            writer.writerow([image_path, image_path])

    with pytest.raises(ValueError):
        PairedImageFolder(str(root), _toTensorTransform(), _toTensorTransform(),
                          segmentation_pairing=str(pairing),
                          shared_augmentation=transforms.RandomHorizontalFlip(p=1.0))


def testTargetAugmentationAffectsOnlyTheTarget(imageFolder: Tuple[Path, List[str]]) -> None:
    """A target-side photometric augmentation must change the target while
    leaving the model input pristine (chroma jitter is target-only by design)."""
    root, _ = imageFolder

    def paintRed(image: Image.Image) -> Image.Image:
        return Image.new("RGB", image.size, color=(255, 0, 0))

    dataset: PairedImageFolder = PairedImageFolder(
        str(root), _toTensorTransform(), _toTensorTransform(), target_augmentation=paintRed)
    reference: PairedImageFolder = PairedImageFolder(
        str(root), _toTensorTransform(), _toTensorTransform())

    model_input, target, _, _ = dataset[0]
    reference_input, _, _, _ = reference[0]
    assert torch.allclose(model_input, reference_input)   # input untouched
    assert torch.allclose(target[0], torch.ones_like(target[0]))   # red channel saturated
    assert torch.allclose(target[1], torch.zeros_like(target[1]))  # green zeroed


def testPathAwareTargetAugmentationReceivesTheImagePath(
        imageFolder: Tuple[Path, List[str]]) -> None:
    """The cluster-version remap picks its correspondence from the source image's cluster, so
    the dataset must hand a path-aware augmentation the file each sample came from."""
    root, filenames = imageFolder

    class Recorder(PathAwareImageTransform):
        def __init__(self) -> None:
            self.seen: List[str] = []

        def __call__(self, image: Image.Image, image_path: str) -> Image.Image:
            self.seen.append(image_path)
            return image

    recorder: Recorder = Recorder()
    dataset: PairedImageFolder = PairedImageFolder(
        str(root), _toTensorTransform(), _toTensorTransform(), target_augmentation=recorder)
    dataset[0]
    dataset[1]

    assert len(recorder.seen) == 2
    assert all(os.path.isfile(path) for path in recorder.seen)
    assert {os.path.basename(path) for path in recorder.seen} <= set(filenames)
    assert recorder.seen[0] != recorder.seen[1]        # each sample gets its own path
