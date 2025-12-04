from typing import Dict, Tuple
from abc import ABC, abstractmethod
import torch
from .visualize import ColorizationVisualizer, SegmentationVisualizer, \
    VisualizationStrategy


class ControlFunction(ABC):
    """Abstract base class for control functions."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        pass


class PhaseAwareControlFunction(ControlFunction):
    """Base class for control functions that handle both train and val phases."""

    def __init__(self, visualizer: VisualizationStrategy):
        self.visualizer = visualizer

    def processPhase(self, phase: str, batch: Tuple,
                     models: Dict[str, torch.nn.Module],
                     epoch: int, diagrams_path: str,
                     config_id: str, model_type: str,
                     launch_number: int) -> None:
        """Process a single phase (train or val). To be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, models: Dict[str, torch.nn.Module],
                 train_batch: Tuple, val_batch: Tuple,
                 epoch: int, diagrams_path: str,
                 config_id: str, model_type: str,
                 launch_number: int) -> None:

        self.processPhase("train", train_batch, models, epoch,
                          diagrams_path, config_id, model_type, launch_number)

        self.processPhase("val", val_batch, models, epoch,
                          diagrams_path, config_id, model_type, launch_number)


class ColorizationControlFunction(PhaseAwareControlFunction):
    """Control function for colorization/reconstruction tasks."""

    def __init__(self, target_channel: str = "RGB", input_channel: str = "RGB"):
        visualizer = ColorizationVisualizer()
        super().__init__(visualizer)
        self.target_channel = target_channel
        self.input_channel = input_channel

    def processPhase(self, phase: str, batch: Tuple,
                     models: Dict[str, torch.nn.Module],
                     epoch: int, diagrams_path: str,
                     config_id: str, model_type: str,
                     launch_number: int) -> None:

        generator = models['generator']
        generator.eval()

        modified, original = batch[0], batch[1]

        with torch.no_grad():
            reconstructed = generator(modified)

        phase_model_type = f"{phase}_{model_type}"

        self.visualizer.visualize(
            original_images=original,
            modified_images=modified,
            reconstructed_images=reconstructed,
            epoch=epoch,
            save_path=diagrams_path,
            model_type=phase_model_type,
            launch_number=launch_number,
            config_id=config_id,
            num_images=8,
            target_channel=self.target_channel,
            input_channel=self.input_channel
        )


class SegmentationControlFunction(PhaseAwareControlFunction):
    """Control function for segmentation tasks."""

    def __init__(self, num_classes: int = 4):
        classColors = {
            1: [0.2, 0.8, 0.2],    # Healthy
            2: [0.9, 0.9, 0.2],    # Partially injured
            3: [0.9, 0.3, 0.3]     # Completely ruptured
        }
        visualizer = SegmentationVisualizer(classColors)
        super().__init__(visualizer)
        self.num_classes = num_classes

    def processPhase(self, phase: str, batch: Tuple,
                     models: Dict[str, torch.nn.Module],
                     epoch: int, diagrams_path: str,
                     config_id: str, model_type: str,
                     launch_number: int) -> None:

        seg_model = models['segformer']
        seg_model.eval()

        volumes, masks = batch[0], batch[1]

        with torch.no_grad():
            preds = seg_model(volumes)

        self.visualizer.visualize(
            volumes=volumes,
            true_masks=masks,
            pred_logits=preds,
            epoch=epoch,
            save_path=diagrams_path,
            phase=phase,
            config_id=config_id,
            model_type=model_type,
            launch_number=launch_number
        )


class ControlFunctionFactory:
    """Factory for creating different types of control functions."""

    @staticmethod
    def createColorizationControl(target_channel: str = "RGB",
                                  input_channel: str = "RGB") -> ColorizationControlFunction:
        """Create a control function for colorization/reconstruction tasks."""
        return ColorizationControlFunction(target_channel, input_channel)

    @staticmethod
    def createSegmentationControl(num_classes: int = 4) -> SegmentationControlFunction:
        """Create a control function for segmentation tasks."""
        return SegmentationControlFunction(num_classes)
