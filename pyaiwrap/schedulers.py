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
    STEP = "step"
    EXPONENTIAL = "exponential"
    POLYWARMUP = "polywarmup"
    MULTI_STEP = "multi_step"


class SchedulerFactory(ABC):
    """Abstract base class for scheduler factories."""

    @abstractmethod
    def createScheduler(self, optimizer, config: Config):
        pass


class CosineWarmRestartsFactory(SchedulerFactory):
    def createScheduler(self, optimizer, config: Config):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["T_0"],
            T_mult=config["T_MULT"],
            eta_min=config["MIN_LR"]
        )


class OneCycleFactory(SchedulerFactory):
    def createScheduler(self, optimizer, config: Config):
        learningRate = config["LEARNING_RATE"]
        trainingDatasetSize = config["TRAINING_DATASET_SIZE"]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learningRate * config["MAX_LR_MULTIPLIER"],
            epochs=config["EPOCHS"],
            steps_per_epoch=trainingDatasetSize,
            pct_start=config["PCT_START"],
            div_factor=config["DIV_FACTOR"],
            final_div_factor=config["FINAL_DIV_FACTOR"]
        )


class CosineFactory(SchedulerFactory):
    def createScheduler(self, optimizer, config: Config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["EPOCHS"],
            eta_min=config["MIN_LR"]
        )


class StepFactory(SchedulerFactory):
    def createScheduler(self, optimizer, config: Config):
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["STEP_SIZE"],
            gamma=config["STEP_GAMMA"]
        )


class ExponentialFactory(SchedulerFactory):
    def createScheduler(self, optimizer, config: Config):
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config["GAMMA"]
        )


class MultiStepFactory(SchedulerFactory):
    """Factory for iteration-based MultiStepLR."""
    def createScheduler(self, optimizer, config: Config):

        milestone_list = []
        decay_start = config["DECAY_START_ITER"]
        decay_step = config["DECAY_STEP_ITER"]
        max_iters = config["MAX_ITERS"]

        current_iter = decay_start
        while current_iter < max_iters:
            milestone_list.append(current_iter)
            current_iter += decay_step

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


class PolyWarmupFactory(SchedulerFactory):
    def createScheduler(self, optimizer, config: Config):
        return PolyWarmupScheduler(
            optimizer,
            warmup_epochs=config["POLY_WARMUP_EPOCHS"],
            total_epochs=config["EPOCHS"],
            base_lr=config["BASE_LR"],
            final_lr=config["FINAL_LR"],
            power=config["POLY_POWER"]
        )


class SchedulerCreator:
    """Factory manager that creates schedulers based on type."""

    _factories = {
        SchedulerType.COSINE_WARM_RESTARTS: CosineWarmRestartsFactory(),
        SchedulerType.ONECYCLE: OneCycleFactory(),
        SchedulerType.COSINE: CosineFactory(),
        SchedulerType.STEP: StepFactory(),
        SchedulerType.EXPONENTIAL: ExponentialFactory(),
        SchedulerType.POLYWARMUP: PolyWarmupFactory(),
        SchedulerType.MULTI_STEP: MultiStepFactory()
    }

    @classmethod
    def createScheduler(cls, scheduler_type: SchedulerType, optimizer, config: Config):
        factory = cls._factories[scheduler_type]
        return factory.createScheduler(optimizer, config)


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
    return SchedulerCreator.createScheduler(scheduler_type, optimizer, config_with_dataset_size)
