from abc import ABC, abstractmethod
from typing import List, Type
from enum import Enum
import torch
from .config import buildNeuralNetworkFromJson
from .neural_network import ConvAttenColorizationNetwork
from pyaiwrap.config import Config


class GeneratorType(Enum):
    """Enum representing different generator types."""
    STANDALONE = "standalone"
    SUBMODULAR = "submodular"


class ModularClassType(Enum):
    """Enum representing different modular class types."""
    CONV_ATTEN_COLORIZATION = "ConvAttenColorizationNetwork"


class GeneratorFactory(ABC):
    """Abstract base class for generator factories."""

    @abstractmethod
    def createGenerator(self, architecture_id: str, submodules: List[str], device: torch.device, config: Config) -> torch.nn.Module:
        pass

    @abstractmethod
    def getDescription(self) -> str:
        pass


class StandaloneGeneratorFactory(GeneratorFactory):
    def createGenerator(self, architecture_id: str, submodules: List[str], device: torch.device, config: Config) -> torch.nn.Module:
        architecture_path = config["ARCHITECTURE_PATH"]
        generator = buildNeuralNetworkFromJson(
            f"{architecture_path}{architecture_id}.json"
        )
        return generator.to(device)

    def getDescription(self) -> str:
        return "Using standalone generator"


class SubmodularGeneratorFactory(GeneratorFactory):
    def createGenerator(self, architecture_id: str, submodules: List[str], device: torch.device, config: Config) -> torch.nn.Module:
        architecture_path = config["ARCHITECTURE_PATH"]
        modular_class_type = config["MODULAR_CLASS"]
        modular_class_type = ModularClassType(modular_class_type)

        trainable_network = buildNeuralNetworkFromJson(
            f"{architecture_path}{architecture_id}.json"
        )

        modular_class = self._getModularClass(modular_class_type)

        generator = modular_class(
            pretrained_models_config=submodules,
            trainable_network=trainable_network
        )
        return generator.to(device)

    def _getModularClass(self, modular_class_type: ModularClassType) -> Type:
        """Get the modular class from enum."""
        class_mapping = {
            ModularClassType.CONV_ATTEN_COLORIZATION: ConvAttenColorizationNetwork,
        }

        return class_mapping[modular_class_type]

    def getDescription(self) -> str:
        return "Using submodular generator with pretrained components"


class GeneratorCreator:
    """Factory manager that creates generators based on type."""

    _factories = {
        GeneratorType.STANDALONE: StandaloneGeneratorFactory(),
        GeneratorType.SUBMODULAR: SubmodularGeneratorFactory()
    }

    @classmethod
    def createGenerator(cls, generator_type: GeneratorType, architecture_id: str,
                        submodules: List[str], device: torch.device, config: Config) -> torch.nn.Module:
        """Create generator based on type."""
        factory = cls._factories[generator_type]
        return factory.createGenerator(architecture_id, submodules, device, config)


def createGenerator(config: Config, device: torch.device) -> torch.nn.Module:
    """
    Create generator based on hyperparameters.

    Args:
        config: Dictionary of hyperparameters
        device: Device to load the model to

    Returns:
        Generator model
    """
    architecture_id = config["ARCHITECTURE_ID"]
    submodules = config["SUBMODULES"]

    if submodules:
        generator_type = GeneratorType.SUBMODULAR
    else:
        generator_type = GeneratorType.STANDALONE

    return GeneratorCreator.createGenerator(generator_type, architecture_id, submodules, device, config)
