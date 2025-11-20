import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
from enum import Enum
import torch
from .config import buildNeuralNetworkFromJson
from .neural_network import ConvAttenColorizationNetwork


def loadHyperparameters(json_path: str) -> Dict[str, Any]:
    """
    Load hyperparameters from a JSON file with enhanced defaults.

    Args:
        json_path (str): Path to the JSON file containing hyperparameters.

    Returns:
        Dict[str, Any]: A dictionary with hyperparameters and their values.
    """
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

    defaults = {
        "BATCH_SIZE": 1,
        "TRAIN_DATA_PATH": "./data/DIV2K_train_LR_bicubic/X4",
        "VALIDATION_DATA_PATH": "./data/DIV2K_valid_LR_bicubic/X4",
        "HYPERPARAMS_ID": "0",
        "ARCHITECTURE_ID": "0",
        "SUBMODULES": {},

        "LEARNING_RATE": 0.0001,
        "WEIGHT_DECAY": 0.01,
        "USE_ADAMW": True,
        "B1": 0.9,
        "B2": 0.999,

        "SCHEDULER_TYPE": "cosine_warm_restarts",  # Options: exponential, cosine_warm_restarts, onecycle, reduce_on_plateau, cosine, step
        "GAMMA": 0.99,
        "MIN_LR": 1e-6,

        "T_0": 30,
        "T_MULT": 2,

        "MAX_LR_MULTIPLIER": 10,
        "PCT_START": 0.1,
        "DIV_FACTOR": 10,
        "FINAL_DIV_FACTOR": 100,

        "LR_REDUCTION_FACTOR": 0.5,
        "LR_PATIENCE": 10,

        "STEP_SIZE": 30,
        "STEP_GAMMA": 0.1,

        "IMAGE_RESIZE": 256,
        "INPUT_CHANNEL": "RGB",
        "OUTPUT_CHANNELS": 3,
        "EPOCHS": 300,
        "DIAGRAMS_DATA_PATH": "./diagrams_data",
        "WEIGHTS_PATH": "./weights",
        "PATIENCE": 30,
        "DIAGRAMS_PATH": "./diagrams",
        "VISUALIZE_EVERY": 5,
        "GRADIENT_CLIP": 1.0,

        "RECON_WEIGHT": 1.0,
        "PERCEPTUAL_WEIGHT": 0.0,
        "USE_LPIPS": False,
        "LPIPS_NET": "alex",
        "COLORFULNESS_WEIGHT": 0.0,
        "COLORFULNESS_TARGET": None,
        "TARGET_CHANNEL": "RGB",
        "TARGET_OUTPUT_CHANNELS": 3,

        "WARMUP_EPOCHS": 2,
        "KL_BETA": 0.01,  # KL divergence weight for VAE,
        "LATENT_DIM": 1024
    }

    for key, default_value in defaults.items():
        hyperparams.setdefault(key, default_value)

    return hyperparams


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