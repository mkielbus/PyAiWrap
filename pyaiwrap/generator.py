from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
from enum import Enum
import torch
from .config import buildNeuralNetworkFromJson
from .neural_network import ConvAttenColorizationNetwork


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
    def createGenerator(self, architecture_id: str, submodules: List[str], device: torch.device, hyperparams: Dict[str, Any]) -> torch.nn.Module:
        pass

    @abstractmethod
    def getDescription(self) -> str:
        pass


class StandaloneGeneratorFactory(GeneratorFactory):
    def createGenerator(self, architecture_id: str, submodules: List[str], device: torch.device, hyperparams: Dict[str, Any]) -> torch.nn.Module:
        architecture_path = hyperparams.get("ARCHITECTURE_PATH", "./network_architectures/generators/")
        generator = buildNeuralNetworkFromJson(
            f"{architecture_path}{architecture_id}.json"
        )
        return generator.to(device)

    def getDescription(self) -> str:
        return "Using standalone generator"


class SubmodularGeneratorFactory(GeneratorFactory):
    def createGenerator(self, architecture_id: str, submodules: List[str], device: torch.device, hyperparams: Dict[str, Any]) -> torch.nn.Module:
        architecture_path = hyperparams.get("ARCHITECTURE_PATH", "./network_architectures/generators/")
        modular_class_type = hyperparams.get("MODULAR_CLASS", ModularClassType.CONV_ATTEN_COLORIZATION)
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
                        submodules: List[str], device: torch.device, hyperparams: Dict[str, Any]) -> torch.nn.Module:
        """Create generator based on type."""
        try:
            factory = cls._factories[generator_type]
        except KeyError:
            factory = StandaloneGeneratorFactory()

        generator = factory.createGenerator(architecture_id, submodules, device, hyperparams)
        print(factory.getDescription())

        return generator


def createGenerator(hyperparams: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Create generator based on hyperparameters.

    Args:
        hyperparams: Dictionary of hyperparameters
        device: Device to load the model to

    Returns:
        Generator model
    """
    architecture_id = hyperparams.get("ARCHITECTURE_ID", "default")
    submodules = hyperparams.get("SUBMODULES", [])

    if submodules:
        generator_type = GeneratorType.SUBMODULAR
    else:
        generator_type = GeneratorType.STANDALONE

    return GeneratorCreator.createGenerator(generator_type, architecture_id, submodules, device, hyperparams)