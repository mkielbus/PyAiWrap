import math
import torch
from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy
from pyaiwrap.config import Config
from torch.optim.lr_scheduler import _LRScheduler
import warnings


class SchedulerType(Enum):
    """Enum representing different scheduler types."""
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    ONECYCLE = "onecycle"
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    STEP = "step"
    EXPONENTIAL = "exponential"
    POLYWARMUP = "polywarmup"
    MULTI_STEP = "multi_step"


class SchedulerCreator(ABC):
    """Abstract base class for scheduler factories."""

    @abstractmethod
    def createScheduler(self, optimizer, config: Config):
        pass


class CosineWarmRestartsCreator(SchedulerCreator):
    def createScheduler(self, optimizer, config: Config):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["T_0"],
            T_mult=config["T_MULT"],
            eta_min=config["MIN_LR"]
        )


class OneCycleCreator(SchedulerCreator):
    def createScheduler(self, optimizer, config: Config):
        learningRate = config["LEARNING_RATE"]
        # total_steps is in epochs because train() steps schedulers once per epoch,
        # not once per batch
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learningRate * config["MAX_LR_MULTIPLIER"],
            total_steps=config["EPOCHS"],
            pct_start=config["PCT_START"],
            div_factor=config["DIV_FACTOR"],
            final_div_factor=config["FINAL_DIV_FACTOR"]
        )


class CosineCreator(SchedulerCreator):
    def createScheduler(self, optimizer, config: Config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["EPOCHS"],
            eta_min=config["MIN_LR"]
        )


class StepCreator(SchedulerCreator):
    def createScheduler(self, optimizer, config: Config):
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["STEP_SIZE"],
            gamma=config["STEP_GAMMA"]
        )


class ExponentialCreator(SchedulerCreator):
    def createScheduler(self, optimizer, config: Config):
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config["GAMMA"]
        )


class MultiStepCreator(SchedulerCreator):
    """Factory for epoch-based MultiStepLR (train() steps schedulers once per epoch)."""
    def createScheduler(self, optimizer, config: Config):

        milestone_list = []
        decay_start = config["DECAY_START_EPOCH"]
        decay_step = config["DECAY_STEP_EPOCHS"]
        max_epochs = config["EPOCHS"]

        current_epoch = decay_start
        while current_epoch < max_epochs:
            milestone_list.append(current_epoch)
            current_epoch += decay_step

        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestone_list,
            gamma=config["DECAY_FACTOR"]
        )


class PolyWarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by polynomial decay.
    As described in SegFormer3D paper:
    - Linear warmup from 4e-6 to 4e-4
    - PolyLR decay after warmup
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 50,
        total_epochs: int = 1000,
        base_lr: float = 4e-6,
        final_lr: float = 4e-4,
        power: float = 0.9,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.power = power

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch <= self.warmup_epochs:
            # Linear warmup: from base_lr to final_lr
            progress = self.last_epoch / self.warmup_epochs
            lr = self.base_lr + (self.final_lr - self.base_lr) * progress
        else:
            # Polynomial decay after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            decay = (1 - progress) ** self.power
            lr = self.final_lr * decay

        return [lr for _ in self.optimizer.param_groups]


class CosineWarmupScheduler(_LRScheduler):
    """Linear warmup to a peak learning rate, then cosine decay to a floor.

    The epoch-based counterpart of PolyWarmupScheduler: train() steps schedulers once per
    epoch, so ``last_epoch`` is an epoch index. For the first ``warmup_epochs`` epochs the
    rate rises linearly from ``base_lr`` to ``peak_lr``; the remaining epochs anneal it from
    ``peak_lr`` down to ``min_lr`` following a half-cosine. Both segments meet continuously
    at ``peak_lr``.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 200,
        base_lr: float = 2e-5,
        peak_lr: float = 2e-4,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for the current epoch."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: from base_lr to peak_lr.
            progress = self.last_epoch / self.warmup_epochs
            lr = self.base_lr + (self.peak_lr - self.base_lr) * progress
        else:
            # Cosine decay after warmup: from peak_lr to min_lr.
            decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
            progress = min(1.0, (self.last_epoch - self.warmup_epochs) / decay_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine

        return [lr for _ in self.optimizer.param_groups]


class CosineWarmupCreator(SchedulerCreator):
    def createScheduler(self, optimizer, config: Config):
        return CosineWarmupScheduler(
            optimizer,
            warmup_epochs=config["COSINE_WARMUP_EPOCHS"],
            total_epochs=config["EPOCHS"],
            base_lr=config["BASE_LR"],
            peak_lr=config["PEAK_LR"],
            min_lr=config["MIN_LR"]
        )


class PolyWarmupCreator(SchedulerCreator):
    def createScheduler(self, optimizer, config: Config):
        return PolyWarmupScheduler(
            optimizer,
            warmup_epochs=config["POLY_WARMUP_EPOCHS"],
            total_epochs=config["EPOCHS"],
            base_lr=config["BASE_LR"],
            final_lr=config["FINAL_LR"],
            power=config["POLY_POWER"]
        )


class SchedulerFactory:
    """Factory manager that creates schedulers based on type."""

    _creators = {
        SchedulerType.COSINE_WARM_RESTARTS: CosineWarmRestartsCreator(),
        SchedulerType.ONECYCLE: OneCycleCreator(),
        SchedulerType.COSINE: CosineCreator(),
        SchedulerType.COSINE_WARMUP: CosineWarmupCreator(),
        SchedulerType.STEP: StepCreator(),
        SchedulerType.EXPONENTIAL: ExponentialCreator(),
        SchedulerType.POLYWARMUP: PolyWarmupCreator(),
        SchedulerType.MULTI_STEP: MultiStepCreator()
    }

    @classmethod
    def createScheduler(cls, scheduler_type: SchedulerType, optimizer, config: Config):
        creator = cls._creators[scheduler_type]
        return creator.createScheduler(optimizer, config)


def createScheduler(optimizer, config: Config, train_loader_len: int):
    """
    Create learning rate scheduler based on hyperparameters.

    Args:
        optimizer: The optimizer to schedule
        config: Dictionary of hyperparameters
        train_loader_len: Length of train loader (steps per epoch)
        epochs: Total number of epochs

    Returns:
        Configured learning rate scheduler
    """
    config_with_dataset_size = deepcopy(config)
    config_with_dataset_size["TRAINING_DATASET_SIZE"] = train_loader_len

    scheduler_type = SchedulerType(config_with_dataset_size["SCHEDULER_TYPE"])
    return SchedulerFactory.createScheduler(scheduler_type, optimizer, config_with_dataset_size)
