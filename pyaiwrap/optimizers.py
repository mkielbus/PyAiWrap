from abc import ABC, abstractmethod
from enum import Enum
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Any, Dict, Iterator, List, Set, Union
from pyaiwrap.config import Config


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"


def createParameterGroups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Split a model's parameters into a decayed and a non-decayed AdamW group.

    Weight decay is meant for the matmul and convolution weights that define what a layer
    computes. Applied to normalisation gains, biases and learned embeddings it instead
    shrinks parameters that carry no capacity of their own, and for an embedding that gets
    only a weak gradient it pulls the weights towards zero faster than training builds them
    up -- a zero-initialised query table then never leaves zero. Everything with ndim <= 1
    (norm gains, every bias) and every nn.Embedding weight is therefore excluded; the
    embeddings need the explicit check because their tables are 2-D.

    Groups that end up empty are dropped, so a model made only of norms and biases still
    yields a usable optimizer.
    """
    embedding_parameter_ids: Set[int] = {
        id(parameter)
        for module in model.modules() if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }

    decay_parameters: List[Parameter] = []
    no_decay_parameters: List[Parameter] = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or id(parameter) in embedding_parameter_ids:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    groups: List[Dict[str, Any]] = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0}
    ]
    return [group for group in groups if group["params"]]


def resolveOptimizerInput(model_or_parameters: Union[nn.Module, Iterator[Parameter]],
                          weight_decay: float,
                          no_decay_groups: bool) -> Any:
    """Turn the caller's argument into whatever AdamW should receive.

    The split has to be opted into per config, not decided by the argument type. A single
    training script serves every model family here, so keying the behaviour off "a module
    was passed" would silently re-tune optimizers for configs that were trained without it,
    and their saved one-group optimizer state would no longer load on resume. With
    `no_decay_groups` off, a module is simply unwrapped into its parameters.
    """
    if not isinstance(model_or_parameters, nn.Module):
        return model_or_parameters
    if no_decay_groups:
        return createParameterGroups(model_or_parameters, weight_decay)
    return model_or_parameters.parameters()


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
        # Adam applies no decay, so a module argument only needs unwrapping, not splitting.
        if isinstance(parameters, nn.Module):
            parameters = parameters.parameters()
        return torch.optim.Adam(
            parameters,
            lr=config["LEARNING_RATE"]
        )

    def getDescription(self, config: Config) -> str:
        return "Using Adam optimizer"


class AdamWOptimizerCreator(OptimizerCreator):
    def createOptimizer(self, parameters, config: Config) -> torch.optim.Optimizer:
        weight_decay = config["WEIGHT_DECAY"]
        no_decay_groups = config.get("NO_DECAY_GROUPS", False)
        return torch.optim.AdamW(
            resolveOptimizerInput(parameters, weight_decay, no_decay_groups),
            lr=config["LEARNING_RATE"],
            weight_decay=weight_decay,
            betas=(
                config["B1"],
                config["B2"]
            )
        )

    def getDescription(self, config: Config) -> str:
        weight_decay = config["WEIGHT_DECAY"]
        scope = ("weights only (norms/biases/embeddings excluded)"
                 if config.get("NO_DECAY_GROUPS", False) else "all parameters")
        return f"Using AdamW optimizer with weight decay: {weight_decay} on {scope}"


class OptimizerFactory:
    """Factory manager that creates optimizers based on type."""

    _creators = {
        OptimizerType.ADAM: AdamOptimizerCreator(),
        OptimizerType.ADAMW: AdamWOptimizerCreator()
    }

    @classmethod
    def createOptimizer(cls, optimizer_type: OptimizerType,
                        parameters: Union[nn.Module, Iterator[Parameter]],
                        config: Config) -> torch.optim.Optimizer:
        """Create optimizer and return as dictionary."""
        creator = cls._creators[optimizer_type]
        return creator.createOptimizer(parameters, config)


def createOptimizer(model_parameters: Union[nn.Module, Iterator[Parameter]],
                    config: Config) -> torch.optim.Optimizer:
    """
    Create optimizers based on hyperparameters.

    Args:
        model_parameters: The generator module, or a bare iterator over its parameters.
            Passing the module lets AdamW exclude norms, biases and embeddings from weight
            decay (see createParameterGroups); an iterator keeps the old single-group form.
        config: Dictionary of hyperparameters

    Returns:
        Dictionary containing optimizers
    """
    optimizer_type = config["OPTIMIZER_TYPE"]
    optimizer_type = OptimizerType(optimizer_type)

    return OptimizerFactory.createOptimizer(optimizer_type, model_parameters, config)
