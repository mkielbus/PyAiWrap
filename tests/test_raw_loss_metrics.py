"""Tests for the unweighted loss terms logged alongside the weighted ones.

Why they exist: the colorization loss logs `perceptual_weight * LPIPS` under the name
`perceptual_loss`, so a quality target expressed against that number silently changes meaning
whenever the weight is retuned. The `*_raw` entries are the terms themselves, comparable
across configs and directly against published LPIPS figures.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

from pyaiwrap.loss import GeneratorColorizationLoss
from pyaiwrap.metrics import GeneratorColorizationMetrics


class RecordingMetrics:
    """Captures whatever the loss accumulates, so the raw/weighted relation can be asserted."""

    def __init__(self) -> None:
        self.records: List[Dict[str, float]] = []

    def accumulate(self, loss_dict: Dict[str, float]) -> None:
        self.records.append(loss_dict)


def buildLoss(recon_weight: float, perceptual_weight: float) -> GeneratorColorizationLoss:
    return GeneratorColorizationLoss(
        reconstruction_loss_fn=nn.L1Loss(),
        recon_weight=recon_weight,
        perceptual_weight=perceptual_weight,
        colorfulness_weight=0.0,
        use_lpips=False,
        device=torch.device("cpu"),
        input_channel="luminance",
        target_channel="RGB"
    )


def runOneBatch(loss_fn: GeneratorColorizationLoss) -> Dict[str, float]:
    torch.manual_seed(0)
    generator: nn.Module = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, padding=1), nn.Sigmoid())
    metrics: RecordingMetrics = RecordingMetrics()
    batch = (torch.rand(2, 1, 16, 16), torch.rand(2, 3, 16, 16))

    with torch.no_grad():
        loss_fn({"generator": generator}, batch, metrics)

    return metrics.records[0]


def testRawReconstructionIsTheUnweightedTerm() -> None:
    record: Dict[str, float] = runOneBatch(buildLoss(recon_weight=0.02, perceptual_weight=0.0))

    assert record["reconstruction_raw"] == pytest.approx(
        record["reconstruction_loss"] / 0.02, rel=1e-6
    )


def testRawReconstructionIsWeightIndependent() -> None:
    """The same predictions must yield the same raw term whatever the weight is set to."""
    first: Dict[str, float] = runOneBatch(buildLoss(recon_weight=1.0, perceptual_weight=0.0))
    second: Dict[str, float] = runOneBatch(buildLoss(recon_weight=0.02, perceptual_weight=0.0))

    assert first["reconstruction_raw"] == pytest.approx(second["reconstruction_raw"], rel=1e-6)
    assert first["reconstruction_loss"] != pytest.approx(second["reconstruction_loss"], rel=1e-6)


def testPerceptualRawIsZeroWhenPerceptualIsDisabled() -> None:
    record: Dict[str, float] = runOneBatch(buildLoss(recon_weight=1.0, perceptual_weight=0.0))

    assert record["perceptual_raw"] == 0.0
    assert record["perceptual_loss"] == 0.0


def testMetricsTrackRawKeys() -> None:
    metrics: GeneratorColorizationMetrics = GeneratorColorizationMetrics(
        use_colorfulness=True, use_perceptual_loss=True
    )
    assert "reconstruction_raw" in metrics.metric_keys
    assert "perceptual_raw" in metrics.metric_keys


def testMetricsOmitPerceptualRawWhenPerceptualIsOff() -> None:
    metrics: GeneratorColorizationMetrics = GeneratorColorizationMetrics(
        use_colorfulness=False, use_perceptual_loss=False
    )
    assert "reconstruction_raw" in metrics.metric_keys
    assert "perceptual_raw" not in metrics.metric_keys


def testHistoryExportToleratesEpochsPredatingAMetric() -> None:
    """A run resumed from a checkpoint written before *_raw existed must still export."""
    metrics: GeneratorColorizationMetrics = GeneratorColorizationMetrics(
        use_colorfulness=False, use_perceptual_loss=True
    )
    metrics.setState({"history": {
        "train": [{"epoch": 1, "total_loss": 1.0, "reconstruction_loss": 0.1,
                   "perceptual_loss": 0.9}],
        "val": []
    }})

    history = metrics.getHistoryLists()
    assert history["train_total_loss"] == [1.0]
    assert len(history["train_perceptual_raw"]) == 1


def testGradientNormIsRecordedBeforeClipping() -> None:
    """A post-clip log reads back as the threshold on every clipped step and says nothing."""
    from pyaiwrap.metrics import GeneratorColorizationMetrics

    metrics = GeneratorColorizationMetrics(use_colorfulness=True, use_perceptual_loss=True,
                                           track_gradient_norm=True)
    assert "gradient_norm" in metrics.metric_keys

    without = GeneratorColorizationMetrics(use_colorfulness=True, use_perceptual_loss=True)
    assert "gradient_norm" not in without.metric_keys
