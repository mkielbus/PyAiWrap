from abc import ABC, abstractmethod
from typing import Dict, Any
from enum import Enum
import torch
from torch.nn.parameter import Parameter
from typing import Iterator


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class OptimizerFactory(ABC):
    """Abstract base class for optimizer factories."""

    @abstractmethod
    def createOptimizer(self, parameters, hyperparams: Dict[str, Any]) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def getDescription(self, hyperparams: Dict[str, Any]) -> str:
        pass


class AdamOptimizerFactory(OptimizerFactory):
    def createOptimizer(self, parameters, hyperparams: Dict[str, Any]) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            parameters,
            lr=hyperparams.get("LEARNING_RATE", 0.0001)
        )

    def getDescription(self, hyperparams: Dict[str, Any]) -> str:
        return "Using Adam optimizer"


class AdamWOptimizerFactory(OptimizerFactory):
    def createOptimizer(self, parameters, hyperparams: Dict[str, Any]) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            parameters,
            lr=hyperparams.get("LEARNING_RATE", 0.0001),
            weight_decay=hyperparams.get("WEIGHT_DECAY", 0.01),
            betas=(
                hyperparams.get("B1", 0.5),
                hyperparams.get("B2", 0.999)
            )
        )

    def getDescription(self, hyperparams: Dict[str, Any]) -> str:
        weight_decay = hyperparams.get("WEIGHT_DECAY", 0.01)
        return f"Using AdamW optimizer with weight decay: {weight_decay}"


class OptimizerCreator:
    """Factory manager that creates optimizers based on type."""

    _factories = {
        OptimizerType.ADAM: AdamOptimizerFactory(),
        OptimizerType.ADAMW: AdamWOptimizerFactory()
    }

    @classmethod
    def createOptimizer(cls, optimizer_type: OptimizerType, parameters,
                        hyperparams: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer and return as dictionary."""
        try:
            factory = cls._factories[optimizer_type]
        except KeyError:
            factory = AdamOptimizerFactory()

        optimizer = factory.createOptimizer(parameters, hyperparams)

        return optimizer


def createOptimizer(model_parameters: Iterator[Parameter],
                    hyperparams: Dict[str, Any]) -> Dict[str, torch.optim.Optimizer]:
    """
    Create optimizers based on hyperparameters.

    Args:
        model_parameters: Generator model parameters
        hyperparams: Dictionary of hyperparameters

    Returns:
        Dictionary containing optimizers
    """
    optimizer_type = hyperparams.get("OPTIMIZER_TYPE", OptimizerType.ADAM.value)
    try:
        optimizer_type = OptimizerType(optimizer_type)
    except ValueError:
        optimizer_type = OptimizerType.ADAM

    return OptimizerCreator.createOptimizer(optimizer_type, model_parameters, hyperparams)
