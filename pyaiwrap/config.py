from typing import Any, Dict, List
import json
from .neural_network import NeuralNetwork
from abc import ABC, abstractmethod
from enum import Enum
import copy


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class MissingDataTypeError(ConfigurationError):
    """Raised when DATA_TYPE is missing from JSON configuration."""

    def __init__(self):
        super().__init__("DATA_TYPE key is required in the configuration JSON file.")


class MissingOutputTypeError(ConfigurationError):
    """Raised when OUTPUT_TYPE is missing from JSON configuration."""

    def __init__(self):
        super().__init__("OUTPUT_TYPE key is required in the configuration JSON file.")


class MissingTrainingTypeError(ConfigurationError):
    """Raised when TRAINING_TYPE is missing from JSON configuration."""

    def __init__(self):
        super().__init__("TRAINING_TYPE key is required in the configuration JSON file.")


class MissingOptimizerTypeError(ConfigurationError):
    """Raised when OPTIMIZER_TYPE is missing from JSON configuration."""

    def __init__(self):
        super().__init__("OPTIMIZER_TYPE key is required in the configuration JSON file.")


class MissingSchedulerTypeError(ConfigurationError):
    """Raised when SCHEDULER_TYPE is missing from JSON configuration."""

    def __init__(self):
        super().__init__("SCHEDULER_TYPE key is required in the configuration JSON file.")


class MissingLossTypesError(ConfigurationError):
    """Raised when LOSS_TYPES is missing from JSON configuration."""

    def __init__(self):
        super().__init__("LOSS_TYPES key is required in the configuration JSON file.")


class MissingModelTypeError(ConfigurationError):
    """Raised when MODEL_TYPE is missing from JSON configuration."""

    def __init__(self):
        super().__init__("MODEL_TYPE key is required in the configuration JSON file.")


class InvalidDataTypeError(ConfigurationError):
    """Raised when DATA_TYPE value cannot be converted to DataType enum."""

    def __init__(self, data_type: str):
        super().__init__(f"Invalid DATA_TYPE value: '{data_type}'. Must be one of: {[e.value for e in DataType]}")


class InvalidOutputTypeError(ConfigurationError):
    """Raised when OUTPUT_TYPE value cannot be converted to OutputType enum."""

    def __init__(self, output_type: str):
        super().__init__(f"Invalid OUTPUT_TYPE value: '{output_type}'. Must be one of: {[e.value for e in OutputType]}")


class InvalidTrainingTypeError(ConfigurationError):
    """Raised when TRAINING_TYPE value cannot be converted to TrainingType enum."""

    def __init__(self, training_type: str):
        super().__init__(f"Invalid TRAINING_TYPE value: '{training_type}'. Must be one of: {[e.value for e in TrainingType]}")


class InvalidOptimizerTypeError(ConfigurationError):
    """Raised when OPTIMIZER_TYPE value cannot be converted to OptimizerType enum."""

    def __init__(self, optimizer_type: str):
        super().__init__(f"Invalid OPTIMIZER_TYPE value: '{optimizer_type}'. Must be one of: {[e.value for e in OptimizerType]}")


class InvalidSchedulerTypeError(ConfigurationError):
    """Raised when SCHEDULER_TYPE value cannot be converted to SchedulerType enum."""

    def __init__(self, scheduler_type: str):
        super().__init__(f"Invalid SCHEDULER_TYPE value: '{scheduler_type}'. Must be one of: {[e.value for e in SchedulerType]}")


class InvalidLossTypeError(ConfigurationError):
    """Raised when a value in LOSS_TYPES cannot be converted to LossType enum."""

    def __init__(self, loss_type: str):
        super().__init__(f"Invalid loss type: '{loss_type}'. Must be one of: {[e.value for e in LossType]}")


class InvalidModelTypeError(ConfigurationError):
    """Raised when MODEL_TYPE value cannot be converted to ModelType enum."""

    def __init__(self, model_type: str):
        super().__init__(f"Invalid MODEL_TYPE value: '{model_type}'. Must be one of: {[e.value for e in ModelType]}")


class InvalidFieldsError(ConfigurationError):
    """Raised when keys in input JSON are invalid."""

    def __init__(self, fields_names: str):
        super().__init__(f"Invalid fields in input json file: '{fields_names}'")


def loadLayersFromJson(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSON file containing an array of layer configs."""
    with open(file_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, list):
        raise TypeError("JSON must contain a list of layer definitions.")
    return data


def buildNeuralNetworkFromJson(file_path: str) -> NeuralNetwork:
    layers = loadLayersFromJson(file_path)
    return NeuralNetwork(layers)


class ConfigCategory(Enum):
    """Enum representing different configuration categories."""
    DATA = "data"
    TRAINING = "training"
    SCHEDULER = "scheduler"
    LOSS = "loss"
    MODEL = "model"
    OUTPUT = "output"


class DataType(Enum):
    """Enum representing different data types."""
    COLORIZATION = "colorization"
    STANDARD = "standard"


class OutputType(Enum):
    """Enum representing different output types."""
    STANDARD = "standard"


class SchedulerType(Enum):
    """Enum representing different scheduler types."""
    EXPONENTIAL = "exponential"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    ONECYCLE = "onecycle"
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    STEP = "step"
    POLYWARMUP = "polywarmup"
    MULTI_STEP = "multi_step"


class LossType(Enum):
    """Enum representing different loss types."""
    RECONSTRUCTION = "reconstruction"
    PERCEPTUAL = "perceptual"
    COLORFULNESS = "colorfulness"
    VAE = "vae"
    SEGMENTATION = "segmentation"


class ModelType(Enum):
    """Enum representing different model types."""
    STANDARD = "standard"
    SUBMODULAR = "submodular"
    VAE = "vae"


class OptimizerType(Enum):
    """Enum representing different optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"


class TrainingType(Enum):
    """Enum representing different training types."""
    STANDARD = "standard"
    GAN = "gan"


class Config:
    """Configuration class that wraps dictionary data with type-safe access."""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Safe get with default value."""
        return self._data.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        """Update configuration with another dictionary."""
        self._data.update(other)

    def toDict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return copy.deepcopy(self._data)

    def copy(self) -> 'Config':
        return Config(copy.deepcopy(self._data))


class ConfigBuilder(ABC):
    """Abstract base class for configuration builders."""

    @abstractmethod
    def getDefaults(self) -> Dict[str, Any]:
        """Get default values for this configuration category."""
        pass

    @abstractmethod
    def getCategory(self) -> ConfigCategory:
        """Get the category this builder handles."""
        pass

    def build(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Build configuration with user params and defaults."""
        defaults = self.getDefaults()
        config = {}

        for key, default_value in defaults.items():
            config[key] = user_params.get(key, default_value)

        return config


class DataConfigBuilderFactory:
    """Factory for creating data-specific configuration builders."""

    @staticmethod
    def createBuilder(data_type: DataType) -> ConfigBuilder:
        """Create appropriate data builder based on type."""
        builder_mapping = {
            DataType.COLORIZATION: ColorizationDataConfigBuilder(),
            DataType.STANDARD: StandardDataConfigBuilder()
        }

        return builder_mapping.get(data_type, ColorizationDataConfigBuilder())


class ColorizationDataConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "DATA_TYPE": "colorization",
            "BATCH_SIZE": 1,
            # DataLoader worker processes. The target-side augmentations (chroma jitter, cluster
            # version remap) run on the CPU per sample, so too few workers starves the GPU;
            # default 2 preserves the historical behaviour. Sensible upper bound is the machine's
            # core count minus one or two.
            "NUM_WORKERS": 2,
            "TRAIN_DATA_PATH": "./data/DIV2K_train_LR_bicubic/X4",
            "VALIDATION_DATA_PATH": "./data/DIV2K_valid_LR_bicubic/X4",
            "IMAGE_RESIZE": 256,
            "INPUT_CHANNEL": "RGB",
            "OUTPUT_CHANNELS": 3,
            "TARGET_CHANNEL": "RGB",
            "TARGET_OUTPUT_CHANNELS": 3,
            # Paired geometric augmentation (train only; applied once per image and
            # shared by the input and target transforms). AUGMENT off keeps the
            # deterministic-resize behaviour unchanged.
            "AUGMENT": False,
            "AUG_FLIP_P": 0.5,
            "AUG_CROP_SCALE_MIN": 0.6,
            # null (default) = the crop keeps the source image's aspect ratio, so squaring it
            # off to IMAGE_RESIZE distorts exactly as much as validation distorts that same
            # image. Setting a band instead randomises the crop's aspect, which distorts
            # training differently from validation (measured: matches ~3% of samples); it is
            # kept only to reproduce runs configured before this default changed, where the
            # band was 0.85-1.18. Both keys must be set together or both left null.
            "AUG_RATIO_MIN": None,
            "AUG_RATIO_MAX": None,
            # Target-side chroma jitter (L5a). AUG_CHROMA_P = 0 disables it, leaving the
            # target unchanged; > 0 scales LAB chroma within the dataset's empirical band.
            # min 1.0 keeps it always additive (never desaturating); max 1.5 stays well
            # inside the p98 band for the vast majority of images (~12% clamped).
            "AUG_CHROMA_P": 0.0,
            "AUG_CHROMA_MIN": 1.0,
            "AUG_CHROMA_MAX": 1.5,
            # Target-side cluster-version remap (L5b): recolour an image to another colour
            # version observed in its own semantic cluster. AUG_REMAP_P = 0 disables it. Needs
            # the Phase 0 analysis artifacts; images missing from them are passed through, so a
            # wrong/missing path degrades to "no augmentation" rather than to corrupt targets.
            # Enable AUGMENT alongside it: target augmentations run after the shared geometric
            # crop, so with AUGMENT on the remap costs ~2 ms/sample instead of ~92 ms full-res.
            # Colours the cluster keeps fixed (sky blue, an all-green backdrop) are frozen by
            # the planner, and whole clusters rejected in the QA pass are blacklisted there.
            "AUG_REMAP_P": 0.0,
            "AUG_REMAP_VERSION_INVENTORY": "",
            "AUG_REMAP_COLOR_SV": "",
            "AUG_REMAP_IMAGE_VERSIONS": "",
            "AUG_REMAP_CLUSTER_NAMES": "",
            # None -> the planner's reviewed defaults (freeze 0.50, support 10).
            "AUG_REMAP_FREEZE_THRESHOLD": None,
            "AUG_REMAP_MIN_SUPPORT": None
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.DATA


class StandardDataConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "DATA_TYPE": "standard",
            "BATCH_SIZE": 1,
            "DATA_PATH": "./data",
            "RESIZE": (32, 128, 128)
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.DATA


class OutputConfigBuilderFactory:
    """Factory for creating output-specific configuration builders."""

    @staticmethod
    def createBuilder(output_type: OutputType) -> ConfigBuilder:
        """Create appropriate output builder based on type."""
        builder_mapping = {
            OutputType.STANDARD: StandardOutputConfigBuilder()
        }

        return builder_mapping.get(output_type, StandardOutputConfigBuilder())


class StandardOutputConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "OUTPUT_TYPE": "standard",
            "DIAGRAMS_DATA_PATH": "./diagrams_data",
            "WEIGHTS_PATH": "./weights",
            "DIAGRAMS_PATH": "./diagrams",
            "VISUALIZE_EVERY": 5
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.OUTPUT


class TrainingConfigBuilderFactory:
    """Factory for creating training-specific configuration builders."""

    @staticmethod
    def createBuilder(training_type: TrainingType) -> ConfigBuilder:
        """Create appropriate training builder based on type."""
        builder_mapping = {
            TrainingType.STANDARD: StandardTrainingConfigBuilder(),
            TrainingType.GAN: GANTrainingConfigBuilder()
        }

        return builder_mapping.get(training_type, StandardTrainingConfigBuilder())


class StandardTrainingConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "TRAINING_TYPE": "standard",
            "EPOCHS": 300,
            "PATIENCE": 30,
            "GRADIENT_CLIP": 1.0,
            # Autocast the generator forward. False = unchanged fp32 behaviour; the loss terms
            # stay fp32 either way (see GeneratorColorizationLoss.mixed_precision).
            "MIXED_PRECISION": False,
            "MIXED_PRECISION_DTYPE": "bfloat16",
            # Exponential moving average of the model weights. USE_EMA off keeps the raw-weight
            # behaviour unchanged. When on, validation/checkpoints use the averaged weights;
            # EMA_DECAY is per optimisation step, EMA_WARMUP_UPDATES ramps the decay from 0 so
            # the random initialisation is forgotten quickly (0 disables the ramp).
            "USE_EMA": False,
            "EMA_DECAY": 0.9999,
            "EMA_WARMUP_UPDATES": 0
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.TRAINING


class GANTrainingConfigBuilder(StandardTrainingConfigBuilder):
    """GAN training builder that extends standard training with GAN-specific parameters."""

    def getDefaults(self) -> Dict[str, Any]:
        parent_defaults = super().getDefaults()
        parent_defaults.update({
            "TRAINING_TYPE": "gan",
            "WARMUP_EPOCHS": 15
        })
        return parent_defaults


class OptimizerConfigBuilderFactory:
    """Factory for creating optimizer-specific configuration builders."""

    @staticmethod
    def createBuilder(optimizer_type: OptimizerType) -> ConfigBuilder:
        """Create appropriate optimizer builder based on type."""
        builder_mapping = {
            OptimizerType.ADAM: AdamOptimizerConfigBuilder(),
            OptimizerType.ADAMW: AdamWOptimizerConfigBuilder()
        }

        return builder_mapping.get(optimizer_type, AdamOptimizerConfigBuilder())


class AdamOptimizerConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "OPTIMIZER_TYPE": "adam",
            "LEARNING_RATE": 0.0001,
            "B1": 0.9,
            "B2": 0.999
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.TRAINING


class AdamWOptimizerConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "OPTIMIZER_TYPE": "adamw",
            "LEARNING_RATE": 0.0001,
            "WEIGHT_DECAY": 0.01,
            "B1": 0.9,
            "B2": 0.999,
            # Opt-in: put norm gains, biases and embeddings in a weight-decay-free group.
            # Defaults to False so existing configs keep the single-group optimizer they
            # were trained with, and their saved optimizer state stays loadable on resume.
            "NO_DECAY_GROUPS": False
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.TRAINING


class SchedulerConfigBuilderFactory:
    """Factory for creating scheduler-specific configuration builders."""

    @staticmethod
    def createBuilder(scheduler_type: SchedulerType) -> ConfigBuilder:
        """Create appropriate scheduler builder based on type."""
        builder_mapping = {
            SchedulerType.EXPONENTIAL: ExponentialSchedulerConfigBuilder(),
            SchedulerType.COSINE_WARM_RESTARTS: CosineWarmRestartsConfigBuilder(),
            SchedulerType.ONECYCLE: OneCycleSchedulerConfigBuilder(),
            SchedulerType.COSINE: CosineSchedulerConfigBuilder(),
            SchedulerType.COSINE_WARMUP: CosineWarmupSchedulerConfigBuilder(),
            SchedulerType.STEP: StepSchedulerConfigBuilder(),
            SchedulerType.POLYWARMUP: PolyWarmupSchedulerConfigBuilder(),
            SchedulerType.MULTI_STEP: MultiStepSchedulerrConfigBuilder()
        }

        return builder_mapping.get(scheduler_type, ExponentialSchedulerConfigBuilder())


class ExponentialSchedulerConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "SCHEDULER_TYPE": "exponential",
            "GAMMA": 0.99,
            "MIN_LR": 1e-6
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.SCHEDULER


class CosineWarmRestartsConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "SCHEDULER_TYPE": "cosine_warm_restarts",
            "T_0": 30,
            "T_MULT": 2,
            "MIN_LR": 1e-6
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.SCHEDULER


class OneCycleSchedulerConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "SCHEDULER_TYPE": "onecycle",
            "MAX_LR_MULTIPLIER": 10,
            "PCT_START": 0.1,
            "DIV_FACTOR": 10,
            "FINAL_DIV_FACTOR": 100
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.SCHEDULER


class CosineSchedulerConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "SCHEDULER_TYPE": "cosine",
            "MIN_LR": 1e-6
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.SCHEDULER


class CosineWarmupSchedulerConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "SCHEDULER_TYPE": "cosine_warmup",
            # Epochs of linear warmup from BASE_LR to PEAK_LR before the cosine decay.
            "COSINE_WARMUP_EPOCHS": 10,
            "BASE_LR": 2e-5,
            "PEAK_LR": 2e-4,
            "MIN_LR": 1e-6
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.SCHEDULER


class StepSchedulerConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "SCHEDULER_TYPE": "step",
            "STEP_SIZE": 30,
            "STEP_GAMMA": 0.1
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.SCHEDULER


class PolyWarmupSchedulerConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "SCHEDULER_TYPE": "polywarmup",
            "POLY_WARMUP_EPOCHS": 50,
            "BASE_LR": 4e-6,
            "FINAL_LR": 4e-4,
            "POLY_POWER": 0.9
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.SCHEDULER


class MultiStepSchedulerrConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "SCHEDULER_TYPE": "multi_step",
            "DECAY_START_EPOCH": 40,
            "DECAY_STEP_EPOCHS": 20,
            "DECAY_FACTOR": 0.5,
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.SCHEDULER


class LossConfigBuilderFactory:
    """Factory for creating loss-specific configuration builders."""

    @staticmethod
    def createBuilder(loss_type: LossType) -> ConfigBuilder:
        """Create appropriate loss builder based on type."""
        builder_mapping = {
            LossType.RECONSTRUCTION: ReconstructionLossConfigBuilder(),
            LossType.PERCEPTUAL: PerceptualLossConfigBuilder(),
            LossType.COLORFULNESS: ColorfulnessLossConfigBuilder(),
            LossType.VAE: VAELossConfigBuilder(),
            LossType.SEGMENTATION: SegmentationLossConfigBuilder()
        }

        return builder_mapping.get(loss_type, ReconstructionLossConfigBuilder())


class ReconstructionLossConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "RECON_WEIGHT": 1.0
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.LOSS


class PerceptualLossConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "PERCEPTUAL_WEIGHT": 0.1,
            "USE_LPIPS": True,
            "LPIPS_NET": "alex"
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.LOSS


class ColorfulnessLossConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "COLORFULNESS_WEIGHT": 0.005,
            "COLORFULNESS_TARGET": None
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.LOSS


class VAELossConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "KL_BETA": 0.01
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.LOSS


class SegmentationLossConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "DICE_WEIGHT": 0.5,
            "CE_WEIGHT": 0.5
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.LOSS


class ModelConfigBuilderFactory:
    """Factory for creating model-specific configuration builders."""

    @staticmethod
    def createBuilder(model_type: ModelType) -> ConfigBuilder:
        """Create appropriate model builder based on type."""
        builder_mapping = {
            ModelType.STANDARD: StandardModelConfigBuilder(),
            ModelType.SUBMODULAR: SubmodularModelConfigBuilder(),
            ModelType.VAE: VAEModelConfigBuilder()
        }

        return builder_mapping.get(model_type, StandardModelConfigBuilder())


class StandardModelConfigBuilder(ConfigBuilder):
    def getDefaults(self) -> Dict[str, Any]:
        return {
            "MODEL_TYPE": "standard",
            "HYPERPARAMS_ID": "0",
            "ARCHITECTURE_ID": "0",
            "ARCHITECTURE_PATH": "./network_architectures/generators/",
            "SUBMODULES": {}
        }

    def getCategory(self) -> ConfigCategory:
        return ConfigCategory.MODEL


class SubmodularModelConfigBuilder(StandardModelConfigBuilder):
    """Submodular model builder that extends standard model with modular class."""

    def getDefaults(self) -> Dict[str, Any]:
        parent_defaults = super().getDefaults()
        parent_defaults.update({
            "MODEL_TYPE": "submodular",
            "MODULAR_CLASS": "ConvAttenColorizationNetwork",
            # None = keep the modular class's per-layout default (see
            # ConvAttenColorizationNetwork._resolveConcatenateInput); True/False overrides it.
            "CONCATENATE_INPUT": None
        })
        return parent_defaults


class VAEModelConfigBuilder(StandardModelConfigBuilder):
    """VAE model builder that extends standard model with VAE-specific parameters."""

    def getDefaults(self) -> Dict[str, Any]:
        parent_defaults = super().getDefaults()
        parent_defaults.update({
            "MODEL_TYPE": "vae",
            "LATENT_DIM": 1024
        })
        return parent_defaults


class ConfigDirector:
    """Director class that orchestrates the configuration building process."""

    def __init__(self):
        pass

    def loadConfig(self, json_path: str) -> Config:
        """
        Load hyperparameters from a JSON file with categorized defaults.

        Args:
            json_path (str): Path to the JSON file containing hyperparameters.

        Returns:
            Config: A Config object with hyperparameters and their values.
        """
        with open(json_path, "r") as f:
            user_params: Dict[str, Any] = json.load(f)

        final_config = Config()

        self._buildDataConfig(user_params, final_config)
        self._buildOutputConfig(user_params, final_config)
        self._buildTrainingConfig(user_params, final_config)
        self._buildOptimizerConfig(user_params, final_config)
        self._buildSchedulerConfig(user_params, final_config)
        self._buildLossConfig(user_params, final_config)
        self._buildModelConfig(user_params, final_config)

        final_config["LOSS_TYPES"] = user_params["LOSS_TYPES"]
        excessive_keys = set(user_params.keys()) - set(final_config.toDict().keys())
        if excessive_keys:
            raise InvalidFieldsError(", ".join(excessive_keys))

        return final_config

    def _buildDataConfig(self, user_params: Dict[str, Any], final_config: Config) -> None:
        """Build data configuration using factory pattern."""
        if "DATA_TYPE" not in user_params:
            raise MissingDataTypeError()

        data_type_str = user_params["DATA_TYPE"]
        try:
            data_type = DataType(data_type_str)
        except ValueError:
            raise InvalidDataTypeError(data_type_str)

        data_builder = DataConfigBuilderFactory.createBuilder(data_type)
        data_config = data_builder.build(user_params)
        final_config.update(data_config)

    def _buildOutputConfig(self, user_params: Dict[str, Any], final_config: Config) -> None:
        """Build output configuration using factory pattern."""
        if "OUTPUT_TYPE" not in user_params:
            raise MissingOutputTypeError()

        output_type_str = user_params["OUTPUT_TYPE"]
        try:
            output_type = OutputType(output_type_str)
        except ValueError:
            raise InvalidOutputTypeError(output_type_str)

        output_builder = OutputConfigBuilderFactory.createBuilder(output_type)
        output_config = output_builder.build(user_params)
        final_config.update(output_config)

    def _buildTrainingConfig(self, user_params: Dict[str, Any], final_config: Config) -> None:
        """Build training configuration using factory pattern."""
        if "TRAINING_TYPE" not in user_params:
            raise MissingTrainingTypeError()

        training_type_str = user_params["TRAINING_TYPE"]
        try:
            training_type = TrainingType(training_type_str)
        except ValueError:
            raise InvalidTrainingTypeError(training_type_str)

        training_builder = TrainingConfigBuilderFactory.createBuilder(training_type)
        training_config = training_builder.build(user_params)
        final_config.update(training_config)

    def _buildOptimizerConfig(self, user_params: Dict[str, Any], final_config: Config) -> None:
        """Build optimizer configuration using factory pattern."""
        if "OPTIMIZER_TYPE" not in user_params:
            raise MissingOptimizerTypeError()

        optimizer_type_str = user_params["OPTIMIZER_TYPE"]
        try:
            optimizer_type = OptimizerType(optimizer_type_str)
        except ValueError:
            raise InvalidOptimizerTypeError(optimizer_type_str)

        optimizer_builder = OptimizerConfigBuilderFactory.createBuilder(optimizer_type)
        optimizer_config = optimizer_builder.build(user_params)
        final_config.update(optimizer_config)

    def _buildSchedulerConfig(self, user_params: Dict[str, Any], final_config: Config) -> None:
        """Build scheduler configuration using factory pattern."""
        if "SCHEDULER_TYPE" not in user_params:
            raise MissingSchedulerTypeError()

        scheduler_type_str = user_params["SCHEDULER_TYPE"]
        try:
            scheduler_type = SchedulerType(scheduler_type_str)
        except ValueError:
            raise InvalidSchedulerTypeError(scheduler_type_str)

        scheduler_builder = SchedulerConfigBuilderFactory.createBuilder(scheduler_type)
        scheduler_config = scheduler_builder.build(user_params)
        final_config.update(scheduler_config)

    def _buildLossConfig(self, user_params: Dict[str, Any], final_config: Config) -> None:
        """Build loss configuration using factory pattern."""
        if "LOSS_TYPES" not in user_params:
            raise MissingLossTypesError()

        loss_types = self._getEnabledLossTypes(user_params)
        for loss_type in loss_types:
            loss_builder = LossConfigBuilderFactory.createBuilder(loss_type)
            loss_config = loss_builder.build(user_params)
            final_config.update(loss_config)

    def _buildModelConfig(self, user_params: Dict[str, Any], final_config: Config) -> None:
        """Build model configuration using factory pattern."""
        if "MODEL_TYPE" not in user_params:
            raise MissingModelTypeError()

        model_type = self._getModelType(user_params)
        model_builder = ModelConfigBuilderFactory.createBuilder(model_type)
        model_config = model_builder.build(user_params)
        final_config.update(model_config)

    def _getEnabledLossTypes(self, user_params: Dict[str, Any]) -> list[LossType]:
        """Determine which loss types are enabled based on LOSS_TYPES from JSON."""
        loss_types_list = user_params["LOSS_TYPES"]
        enabled_loss_types = []

        for loss_type_str in loss_types_list:
            try:
                loss_type = LossType(loss_type_str.lower())
                enabled_loss_types.append(loss_type)
            except ValueError:
                raise InvalidLossTypeError(loss_type_str)

        return enabled_loss_types

    def _getModelType(self, user_params: Dict[str, Any]) -> ModelType:
        """Determine model type based on MODEL_TYPE from JSON."""
        model_type_str = user_params["MODEL_TYPE"]
        try:
            return ModelType(model_type_str.lower())
        except ValueError:
            raise InvalidModelTypeError(model_type_str)


def loadConfig(json_path: str) -> Config:
    """
    Load hyperparameters from a JSON file with enhanced defaults.

    Args:
        json_path (str): Path to the JSON file containing hyperparameters.

    Returns:
        Config: A Config object with hyperparameters and their values.
    """
    hyperparams_director = ConfigDirector()
    return hyperparams_director.loadConfig(json_path)
