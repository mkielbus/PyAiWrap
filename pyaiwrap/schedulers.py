import torch
from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy
from pyaiwrap.config import Config


class SchedulerType(Enum):
    """Enum representing different scheduler types."""
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    ONECYCLE = "onecycle"
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"


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


class SchedulerCreator:
    """Factory manager that creates schedulers based on type."""

    _factories = {
        SchedulerType.COSINE_WARM_RESTARTS: CosineWarmRestartsFactory(),
        SchedulerType.ONECYCLE: OneCycleFactory(),
        SchedulerType.COSINE: CosineFactory(),
        SchedulerType.STEP: StepFactory(),
        SchedulerType.EXPONENTIAL: ExponentialFactory()
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
