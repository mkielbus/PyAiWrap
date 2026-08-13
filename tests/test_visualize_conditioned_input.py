"""Tests that conditioning channels never reach the visualiser's image conversions.

This pins a bug that killed a real run: with segmentation masks the model input is
[luminance, boundary, area], and the luminance conversion tripled all three channels into a
9-channel "grayscale", which then would not concatenate with the 3-channel target. The
visualiser only runs every VISUALIZE_EVERY epochs, so the crash arrived five epochs in, and it
runs *before* the epoch's checkpoint is written, so the epoch in progress was lost with it.

Naming/style follows the project convention (see CLAUDE.md).
"""
from pathlib import Path
from typing import List

import pytest
import torch

from pyaiwrap.visualize import ColorizationVisualizer

BATCH: int = 2
SIZE: int = 8


def _visualise(visualizer: ColorizationVisualizer, modified: torch.Tensor, save_path: Path,
               input_channel: str = "luminance", target_channel: str = "RGB") -> None:
    visualizer.visualize(
        original_images=torch.rand(BATCH, 3, SIZE, SIZE),
        modified_images=modified,
        reconstructed_images=torch.rand(BATCH, 3, SIZE, SIZE),
        epoch=5,
        save_path=str(save_path),
        model_type="custom",
        launch_number="0",
        config_id="test",
        input_channel=input_channel,
        target_channel=target_channel,
    )


def testMaskConditionedInputVisualises(tmp_path: Path) -> None:
    """[luminance, boundary, area] must render, not raise."""
    visualizer: ColorizationVisualizer = ColorizationVisualizer()
    _visualise(visualizer, torch.rand(BATCH, 3, SIZE, SIZE), tmp_path)

    written: List[Path] = list(tmp_path.glob("*.png"))
    assert len(written) == 1


@pytest.mark.parametrize("mask_channels", [1, 2, 5])
def testAnyConditioningWidthVisualises(tmp_path: Path, mask_channels: int) -> None:
    """label_id is 1 channel, boundary_area 2, boundary_area_hash 2 + n."""
    visualizer: ColorizationVisualizer = ColorizationVisualizer()
    _visualise(visualizer, torch.rand(BATCH, 1 + mask_channels, SIZE, SIZE),
               tmp_path / str(mask_channels))

    assert list((tmp_path / str(mask_channels)).glob("*.png"))


def testOnlyTheLuminanceIsDrawn(tmp_path: Path) -> None:
    """The rendered grey must be the luminance channel, not a blend with the mask."""
    visualizer: ColorizationVisualizer = ColorizationVisualizer()
    luminance: torch.Tensor = torch.full((BATCH, 1, SIZE, SIZE), 0.25)
    conditioned: torch.Tensor = torch.cat([luminance, torch.ones(BATCH, 2, SIZE, SIZE)], dim=1)

    prepared_plain = visualizer._prepareImages(
        torch.rand(BATCH, 3, SIZE, SIZE), luminance,
        torch.rand(BATCH, 3, SIZE, SIZE), BATCH, "luminance")
    prepared_conditioned = visualizer._prepareImages(
        torch.rand(BATCH, 3, SIZE, SIZE), conditioned,
        torch.rand(BATCH, 3, SIZE, SIZE), BATCH, "luminance")

    assert torch.equal(prepared_plain["modified"], prepared_conditioned["modified"])


def testUnconditionedInputIsUntouched(tmp_path: Path) -> None:
    """The pre-existing path must be byte-identical: this fix is invisible without masks."""
    visualizer: ColorizationVisualizer = ColorizationVisualizer()
    modified: torch.Tensor = torch.rand(BATCH, 1, SIZE, SIZE)

    prepared = visualizer._prepareImages(torch.rand(BATCH, 3, SIZE, SIZE), modified,
                                         torch.rand(BATCH, 3, SIZE, SIZE), BATCH, "luminance")
    assert torch.equal(prepared["modified"], modified)


def testRgbInputKeepsAllThreeChannels(tmp_path: Path) -> None:
    """Trimming is by the declared input type, so an RGB input loses nothing."""
    visualizer: ColorizationVisualizer = ColorizationVisualizer()
    modified: torch.Tensor = torch.rand(BATCH, 3, SIZE, SIZE)

    prepared = visualizer._prepareImages(torch.rand(BATCH, 3, SIZE, SIZE), modified,
                                         torch.rand(BATCH, 3, SIZE, SIZE), BATCH, "RGB")
    assert torch.equal(prepared["modified"], modified)


def testUnknownChannelTypeIsPassedThrough(tmp_path: Path) -> None:
    """An unrecognised input type must never be the reason a training run dies."""
    visualizer: ColorizationVisualizer = ColorizationVisualizer()
    modified: torch.Tensor = torch.rand(BATCH, 4, SIZE, SIZE)

    prepared = visualizer._prepareImages(torch.rand(BATCH, 3, SIZE, SIZE), modified,
                                         torch.rand(BATCH, 3, SIZE, SIZE), BATCH, "something")
    assert torch.equal(prepared["modified"], modified)


def testAbTargetWithConditionedInputVisualises(tmp_path: Path) -> None:
    """The AB path pairs the input as the L channel, so it breaks the same way if untrimmed."""
    visualizer: ColorizationVisualizer = ColorizationVisualizer()
    visualizer.visualize(
        original_images=torch.rand(BATCH, 2, SIZE, SIZE) * 100 - 50,
        modified_images=torch.rand(BATCH, 3, SIZE, SIZE),
        reconstructed_images=torch.rand(BATCH, 2, SIZE, SIZE) * 100 - 50,
        epoch=5, save_path=str(tmp_path), model_type="custom", launch_number="0",
        config_id="test", input_channel="luminance", target_channel="AB",
    )

    assert list(tmp_path.glob("*.png"))
