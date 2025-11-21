from abc import ABC, abstractmethod
from enum import Enum
import torch
from torch.nn.parameter import Parameter
from typing import Iterator
from pyaiwrap.config import Config


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class OptimizerFactory(ABC):
    """Abstract base class for optimizer factories."""

    @abstractmethod
    def createOptimizer(self, parameters, config: Config) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def getDescription(self, config: Config) -> str:
        pass


class AdamOptimizerFactory(OptimizerFactory):
    def createOptimizer(self, parameters, config: Config) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            parameters,
            lr=config["LEARNING_RATE"]
        )

    def getDescription(self, config: Config) -> str:
        return "Using Adam optimizer"


class AdamWOptimizerFactory(OptimizerFactory):
    def createOptimizer(self, parameters, config: Config) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            parameters,
            lr=config["LEARNING_RATE"],
            weight_decay=config["WEIGHT_DECAY"],
            betas=(
                config["B1"],
                config["B2"]
            )
        )

    def getDescription(self, config: Config) -> str:
        weight_decay = config["WEIGHT_DECAY"]
        return f"Using AdamW optimizer with weight decay: {weight_decay}"


class OptimizerCreator:
    """Factory manager that creates optimizers based on type."""

    _factories = {
        OptimizerType.ADAM: AdamOptimizerFactory(),
        OptimizerType.ADAMW: AdamWOptimizerFactory()
    }

    @classmethod
    def createOptimizer(cls, optimizer_type: OptimizerType, parameters: Iterator[Parameter],
                        config: Config) -> torch.optim.Optimizer:
        """Create optimizer and return as dictionary."""
        factory = cls._factories[optimizer_type]
        return factory.createOptimizer(parameters, config)


def createOptimizer(model_parameters: Iterator[Parameter],
                    config: Config) -> torch.optim.Optimizer:
    """
    Create optimizers based on hyperparameters.

    Args:
        model_parameters: Generator model parameters
        config: Dictionary of hyperparameters

    Returns:
        Dictionary containing optimizers
    """
    optimizer_type = config["OPTIMIZER_TYPE"]
    optimizer_type = OptimizerType(optimizer_type)

    return OptimizerCreator.createOptimizer(optimizer_type, model_parameters, config)
