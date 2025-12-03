from abc import ABC, abstractmethod
from enum import Enum
import torch
from torch.nn.parameter import Parameter
from typing import Iterator
from pyaiwrap.config import Config


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class OptimizerCreator(ABC):
    """Abstract base class for optimizer factories."""

    @abstractmethod
    def createOptimizer(self, parameters, config: Config) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def getDescription(self, config: Config) -> str:
        pass


class AdamOptimizerCreator(OptimizerCreator):
    def createOptimizer(self, parameters, config: Config) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            parameters,
            lr=config["LEARNING_RATE"]
        )

    def getDescription(self, config: Config) -> str:
        return "Using Adam optimizer"


class AdamWOptimizerCreator(OptimizerCreator):
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


class OptimizerFactory:
    """Factory manager that creates optimizers based on type."""

    _creators = {
        OptimizerType.ADAM: AdamOptimizerCreator(),
        OptimizerType.ADAMW: AdamWOptimizerCreator()
    }

    @classmethod
    def createOptimizer(cls, optimizer_type: OptimizerType, parameters: Iterator[Parameter],
                        config: Config) -> torch.optim.Optimizer:
        """Create optimizer and return as dictionary."""
        creator = cls._creators[optimizer_type]
        return creator.createOptimizer(parameters, config)


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

    return OptimizerFactory.createOptimizer(optimizer_type, model_parameters, config)
