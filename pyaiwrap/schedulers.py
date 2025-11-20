import torch
from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy


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
    def createScheduler(self, optimizer, hyperparams: Dict[str, Any]):
        pass


class CosineWarmRestartsFactory(SchedulerFactory):
    def createScheduler(self, optimizer, hyperparams: Dict[str, Any]):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hyperparams.get("T_0", 30),
            T_mult=hyperparams.get("T_MULT", 2),
            eta_min=hyperparams.get("MIN_LR", 1e-6)
        )


class OneCycleFactory(SchedulerFactory):
    def createScheduler(self, optimizer, hyperparams: Dict[str, Any]):
        learningRate = hyperparams.get("LEARNING_RATE", 0.0001)
        trainingDatasetSize = hyperparams.get("TRAINING_DATASET_SIZE", 1)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learningRate * hyperparams.get("MAX_LR_MULTIPLIER", 10),
            epochs=hyperparams.get("EPOCHS", 100),
            steps_per_epoch=trainingDatasetSize,
            pct_start=hyperparams.get("PCT_START", 0.1),
            div_factor=hyperparams.get("DIV_FACTOR", 10),
            final_div_factor=hyperparams.get("FINAL_DIV_FACTOR", 100)
        )


class CosineFactory(SchedulerFactory):
    def createScheduler(self, optimizer, hyperparams: Dict[str, Any]):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hyperparams.get("EPOCHS", 100),
            eta_min=hyperparams.get("MIN_LR", 1e-6)
        )


class StepFactory(SchedulerFactory):
    def createScheduler(self, optimizer, hyperparams: Dict[str, Any]):
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=hyperparams.get("STEP_SIZE", 30),
            gamma=hyperparams.get("STEP_GAMMA", 0.1)
        )


class ExponentialFactory(SchedulerFactory):
    def createScheduler(self, optimizer, hyperparams: Dict[str, Any]):
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=hyperparams.get("GAMMA", 0.99)
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
    def createScheduler(cls, schedulerType: str, optimizer, hyperparams: Dict[str, Any]):
        try:
            schedulerEnum = SchedulerType(schedulerType)
            factory = cls._factories[schedulerEnum]
        except (KeyError, ValueError):
            factory = ExponentialFactory()
        return factory.createScheduler(optimizer, hyperparams)


def createScheduler(optimizer, hyperparams: Dict[str, Any], train_loader_len: int):
    """
    Create learning rate scheduler based on hyperparameters.

    Args:
        optimizer: The optimizer to schedule
        hyperparams: Dictionary of hyperparameters
        train_loader_len: Length of train loader (steps per epoch)
        epochs: Total number of epochs

    Returns:
        Configured learning rate scheduler
    """
    hyperparams_with_dataset_size = deepcopy(hyperparams)
    hyperparams_with_dataset_size["TRAINING_DATASET_SIZE"] = train_loader_len

    schedulerType = hyperparams_with_dataset_size.get("SCHEDULER_TYPE", "exponential")
    return SchedulerCreator.createScheduler(schedulerType, optimizer, hyperparams_with_dataset_size)
