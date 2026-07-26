"""Tests for CosineWarmupScheduler.

Epoch-indexed (train() steps schedulers once per epoch): a linear warmup from base_lr to
peak_lr over warmup_epochs, then a half-cosine decay from peak_lr to min_lr over the rest.
The two segments must meet continuously at peak_lr and the decay must be monotonic.

Naming/style follows the project convention (see CLAUDE.md).
"""
import math
from typing import List

import torch

from pyaiwrap.schedulers import CosineWarmupScheduler

BASE_LR: float = 2e-5
PEAK_LR: float = 2e-4
MIN_LR: float = 1e-6
WARMUP_EPOCHS: int = 10
TOTAL_EPOCHS: int = 200


def _collectSchedule() -> List[float]:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=PEAK_LR)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=TOTAL_EPOCHS,
        base_lr=BASE_LR,
        peak_lr=PEAK_LR,
        min_lr=MIN_LR
    )
    learning_rates: List[float] = []
    for _ in range(TOTAL_EPOCHS):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return learning_rates


def testWarmupStartsAtBaseLr() -> None:
    assert math.isclose(_collectSchedule()[0], BASE_LR, rel_tol=1e-9)


def testWarmupIsLinear() -> None:
    lrs = _collectSchedule()
    # Epoch 5 is halfway through a 10-epoch warmup: LR halfway between base and peak.
    expected = BASE_LR + (PEAK_LR - BASE_LR) * (5 / WARMUP_EPOCHS)
    assert math.isclose(lrs[5], expected, rel_tol=1e-9)


def testPeakReachedAtWarmupBoundary() -> None:
    lrs = _collectSchedule()
    assert math.isclose(lrs[WARMUP_EPOCHS], PEAK_LR, rel_tol=1e-9)


def testDecayMonotonicAfterWarmup() -> None:
    lrs = _collectSchedule()
    decay_phase = lrs[WARMUP_EPOCHS:]
    for earlier, later in zip(decay_phase, decay_phase[1:]):
        assert later <= earlier + 1e-12


def testFinalLrApproachesMin() -> None:
    lrs = _collectSchedule()
    # Last collected LR is epoch TOTAL_EPOCHS-1, one step short of the min floor.
    assert lrs[-1] < BASE_LR
    assert lrs[-1] >= MIN_LR
