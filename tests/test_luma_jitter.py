"""Tests for the input-side tone jitter and its wiring through PairedImageFolder.

The property that makes this augmentation safe is that it touches the INPUT only: the target
keeps its true colours, so the mapping being learnt is unchanged and only the cue the network
reads becomes less exact. The tests pin that separation, the parameter bounds, and the fact
that input and target augmentations are sampled independently.

Naming/style follows the project convention (see CLAUDE.md).
"""
import random
from pathlib import Path
from typing import Callable, List, Tuple

import pytest
import torch
from PIL import Image
from torchvision import transforms

from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.transforms import LumaJitter, createLumaJitter


def buildImageFolder(tmp_path: Path, count: int = 4) -> str:
    """An ImageFolder-shaped directory of deterministic mid-tone colour images."""
    class_directory: Path = tmp_path / "images"
    class_directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image: Image.Image = Image.new("RGB", (32, 32), color=(120 + index, 90, 60))
        image.save(class_directory / f"img_{index}.png")
    return str(tmp_path)


def makeTransform() -> Callable:
    return transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])


def testJitterRejectsInvalidBounds() -> None:
    with pytest.raises(ValueError, match="probability"):
        LumaJitter(probability=1.5)
    with pytest.raises(ValueError, match="gamma_min"):
        LumaJitter(gamma_min=1.2, gamma_max=0.9)
    with pytest.raises(ValueError, match="contrast"):
        LumaJitter(contrast=1.0)
    with pytest.raises(ValueError, match="brightness"):
        LumaJitter(brightness=-0.1)


def testJitterIsIdentityAtZeroProbability() -> None:
    jitter: LumaJitter = createLumaJitter(probability=0.0)
    image: Image.Image = Image.new("RGB", (8, 8), color=(100, 110, 120))
    assert jitter(image) is image


def testJitterChangesThePixelsWhenApplied() -> None:
    random.seed(0)
    jitter: LumaJitter = createLumaJitter(probability=1.0)
    image: Image.Image = Image.new("RGB", (8, 8), color=(100, 110, 120))

    jittered: Image.Image = jitter(image)
    assert list(jittered.getdata()) != list(image.getdata())


def testJitterIsIdentityWithNeutralParameters() -> None:
    """gamma 1, no contrast and no brightness range must leave the image untouched."""
    random.seed(0)
    jitter: LumaJitter = createLumaJitter(probability=1.0, gamma_min=1.0, gamma_max=1.0,
                                          contrast=0.0, brightness=0.0)
    image: Image.Image = Image.new("RGB", (8, 8), color=(100, 110, 120))
    assert list(jitter(image).getdata()) == list(image.getdata())


def testJitterStaysWithinAModestBand() -> None:
    """The perturbation must stay mild -- it is a regulariser, not a domain shift."""
    random.seed(0)
    jitter: LumaJitter = createLumaJitter(probability=1.0)
    image: Image.Image = Image.new("RGB", (8, 8), color=(128, 128, 128))

    to_tensor = transforms.ToTensor()
    original: torch.Tensor = to_tensor(image)
    for _ in range(64):
        deviation: float = (to_tensor(jitter(image)) - original).abs().max().item()
        assert deviation < 0.25


def testJitterRejectsNonPilInput() -> None:
    with pytest.raises(TypeError, match="expects a PIL image"):
        createLumaJitter(probability=1.0)(torch.rand(3, 8, 8))


def testDatasetLeavesTargetUntouchedWhenInputIsJittered(tmp_path: Path) -> None:
    root: str = buildImageFolder(tmp_path)
    transform: Callable = makeTransform()

    plain: PairedImageFolder = PairedImageFolder(
        root, input_transform=transform, target_transform=transform
    )
    jittered: PairedImageFolder = PairedImageFolder(
        root, input_transform=transform, target_transform=transform,
        input_augmentation=createLumaJitter(probability=1.0)
    )

    random.seed(0)
    jittered_input, jittered_target = jittered[0][0], jittered[0][1]
    plain_input, plain_target = plain[0][0], plain[0][1]

    assert torch.equal(plain_target, jittered_target), "target must be untouched"
    assert not torch.equal(plain_input, jittered_input), "input must be perturbed"


def testDatasetWithoutInputAugmentationIsUnchanged(tmp_path: Path) -> None:
    root: str = buildImageFolder(tmp_path)
    transform: Callable = makeTransform()

    without: PairedImageFolder = PairedImageFolder(
        root, input_transform=transform, target_transform=transform
    )
    with_none: PairedImageFolder = PairedImageFolder(
        root, input_transform=transform, target_transform=transform, input_augmentation=None
    )

    assert torch.equal(without[0][0], with_none[0][0])
    assert torch.equal(without[0][1], with_none[0][1])


def testInputAndTargetAugmentationsCompose(tmp_path: Path) -> None:
    """Both branches may be augmented at once, each from the same shared geometric view."""
    root: str = buildImageFolder(tmp_path)
    transform: Callable = makeTransform()

    def darkenTarget(image: Image.Image) -> Image.Image:
        return Image.eval(image, lambda value: max(0, value - 20))

    dataset: PairedImageFolder = PairedImageFolder(
        root, input_transform=transform, target_transform=transform,
        input_augmentation=createLumaJitter(probability=1.0),
        target_augmentation=darkenTarget
    )
    reference: PairedImageFolder = PairedImageFolder(
        root, input_transform=transform, target_transform=transform
    )

    random.seed(0)
    model_input, target = dataset[0][0], dataset[0][1]
    assert not torch.equal(model_input, reference[0][0])
    assert not torch.equal(target, reference[0][1])
    assert not torch.equal(model_input, target)
