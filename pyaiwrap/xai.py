from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime
from captum.attr import Lime, Saliency, LayerGradCam, LayerAttribution
import torch.nn.functional as F

ActivationDict = Dict[str, torch.Tensor]
LayerNameList = List[str]
BatchType = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


class ActivationHook:
    """Hook to capture activations from a specific layer."""

    def __init__(self, layer_name: str):
        self._layer_name = layer_name
        self._activation = None
        self._hook_handle = None

    def _hookFunction(self, module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Hook callback function."""
        self._activation = output_tensor.detach().clone()

    def attach(self, module: nn.Module):
        """Attach hook to module."""
        self._hook_handle = module.register_forward_hook(self._hookFunction)
        return self

    def detach(self):
        """Remove hook from module."""
        if self._hook_handle:
            self._hook_handle.remove()

    def getActivation(self) -> Optional[torch.Tensor]:
        """Get captured activation."""
        return self._activation

    def getLayerName(self) -> str:
        """Get layer name."""
        return self._layer_name


class ActivationConcatenator:
    """Helper class to concatenate activations across batches."""

    @staticmethod
    def concatenateActivations(activations_list: List[ActivationDict]) -> ActivationDict:
        """Concatenate activations across batches."""
        if not activations_list:
            return {}

        result = {}
        layer_names = activations_list[0].keys()

        for layer_name in layer_names:
            layer_activations = [batch_acts[layer_name] for batch_acts in activations_list
                                 if layer_name in batch_acts]
            if layer_activations:
                result[layer_name] = torch.cat(layer_activations, dim=0)

        return result


class ActivationExtractionCommand(ABC):
    """Command interface for extracting activations from models."""

    @abstractmethod
    def extract(self,
                model: nn.Module,
                dataloader: DataLoader,
                layer_names: LayerNameList) -> ActivationDict:
        """Extract activations from specified layers."""
        pass

    @abstractmethod
    def getName(self) -> str:
        """Get the name of this extraction command."""
        pass


class InputModificationCommand(ABC):
    """Command interface for modifying input tensors."""

    @abstractmethod
    def modify(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply modification to the tensor."""
        pass

    @abstractmethod
    def getDescription(self) -> str:
        """Get description of the modification."""
        pass


class PyTorchHookExtractionCommand(ActivationExtractionCommand):
    """Extract activations from PyTorch models using hooks."""

    def __init__(self):
        self._hooks: List[ActivationHook] = []

    def extract(self,
                model: nn.Module,
                dataloader: DataLoader | List[torch.Tensor],
                layer_names: LayerNameList) -> ActivationDict:
        """Extract activations using forward hooks."""
        self._setupHooks(model, layer_names)
        activations_list = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_tensor = self._extractInputFromBatch(batch)
                _ = model(input_tensor)

                batch_activations = self._collectBatchActivations()
                activations_list.append(batch_activations)

        self._clearHooks()
        return ActivationConcatenator.concatenateActivations(activations_list)

    def _setupHooks(self, model: nn.Module, layer_names: LayerNameList):
        """Setup hooks for specified layers."""
        for name, module in model.named_modules():
            if name and name in layer_names:
                hook = ActivationHook(name).attach(module)
                self._hooks.append(hook)

    def _extractInputFromBatch(self, batch: BatchType) -> torch.Tensor:
        """Extract input tensor from batch."""
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def _collectBatchActivations(self) -> ActivationDict:
        """Collect activations from current batch."""
        activations = {}
        for hook in self._hooks:
            activation = hook.getActivation()
            if activation is not None:
                activations[hook.getLayerName()] = activation
        return activations

    def _clearHooks(self):
        """Clear all hooks."""
        for hook in self._hooks:
            hook.detach()
        self._hooks.clear()

    def getName(self) -> str:
        return "PyTorchHookExtractionCommand"


class ActivationStorage:
    """Handle saving and loading of activations."""

    @staticmethod
    def saveActivations(activations: ActivationDict,
                        file_path: Union[str, Path],
                        metadata: Optional[Dict] = None):
        """Save activations to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'activations': activations,
            'metadata': metadata or {}
        }

        torch.save(save_data, file_path)

    @staticmethod
    def loadActivations(file_path: Union[str, Path]) -> Dict:
        """Load activations from file."""
        return torch.load(file_path, weights_only=False)


@dataclass
class SensitivityAnalysisResult:
    """Container for sensitivity analysis results."""

    layer_names: List[str]
    sensitivity_scores: List[float]
    modification_description: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def getSortedResults(self) -> List[Tuple[str, float]]:
        """Get results sorted by sensitivity."""
        return sorted(zip(self.layer_names, self.sensitivity_scores),
                      key=lambda x: x[1], reverse=True)


class SensitivityVisualizer:
    """Visualize sensitivity analysis results."""

    @staticmethod
    def plotSensitivityBarChart(result: SensitivityAnalysisResult,
                                save_path: Optional[Union[str, Path]] = None):
        """Create bar chart of sensitivity scores."""
        sorted_results = result.getSortedResults()
        layers, scores = zip(*sorted_results) if sorted_results else ([], [])

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(layers)), scores, color='skyblue')
        plt.xlabel('Layers')
        plt.ylabel('Mean Absolute Activation Difference')
        plt.title(f'Sensitivity Analysis: {result.modification_description}')
        plt.xticks(range(len(layers)), layers, rotation=45, ha='right')

        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{score:.4f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()


class XAIExplanationMethod(ABC):
    """Base class for XAI explanation methods."""

    @abstractmethod
    def explain(self,
                model: nn.Module,
                input_tensor: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Generate explanations for the input."""
        pass

    @abstractmethod
    def getName(self) -> str:
        """Get name of the explanation method."""
        pass


class KneeMRIDatasetExplainer(XAIExplanationMethod):
    """Base class for explainers tailored to knee MRI datasets."""

    def _findMostCommonKneeClass(self, model, input_tensor):
        """Find most common non-background class (called ONCE)."""
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)

            knee_mask = probs.argmax(dim=1) != 0

            if knee_mask.any():
                knee_pred = probs.argmax(dim=1)[knee_mask]
                unique, counts = torch.unique(knee_pred, return_counts=True)

                if len(unique) > 0:
                    target_class = unique[counts.argmax()].item()

                    return target_class

        return 0

    def _getKneeClassProbabilities(self, model, x, num_classes=None, use_gradients: bool = False):
        """Get probabilities for ALL classes in knee region."""
        if use_gradients:
            return self._getProbabilities(model, x, num_classes)
        with torch.no_grad():
            return self._getProbabilities(model, x, num_classes)

    def _getProbabilities(self, model, x, num_classes=None):
        output = model(x)
        probs = torch.softmax(output, dim=1)  # B x C x D x H x W

        if num_classes is None:
            num_classes = probs.shape[1]

        # Knee mask (non-background)
        knee_mask = probs.argmax(dim=1) != 0  # B x D x H x W

        if not knee_mask.any():
            # No knee tissue - return zeros for all classes
            return torch.zeros((x.shape[0], num_classes), device=x.device)

        # For EACH class, compute average probability in knee region
        batch_results = []
        for b in range(x.shape[0]):
            class_probs = []
            for c in range(num_classes):
                # Get probabilities for this class in knee region
                class_probs_in_knee = probs[b, c][knee_mask[b]]

                if class_probs_in_knee.numel() > 0:
                    avg_prob = class_probs_in_knee.mean()
                else:
                    avg_prob = torch.tensor(0.0, device=x.device)

                class_probs.append(avg_prob)

            # Shape: [num_classes]
            batch_results.append(torch.stack(class_probs))

        # Result: B x C tensor
        return torch.stack(batch_results, dim=0)


class LIMEExplainer(KneeMRIDatasetExplainer):
    """LIME explanation for medical image segmentation."""

    def __init__(self,
                 n_samples: int = 100,
                 batch_size: int = 4,
                 segmentation_mode: bool = True):
        """
        Args:
            n_samples: Number of samples for LIME
            kernel_width: Kernel width for LIME
            batch_size: Batch size for processing
            segmentation_mode: If True, expects 4D/5D inputs (B x C x D x H x W)
        """
        self._n_samples = n_samples
        self._batch_size = batch_size
        self._segmentation_mode = segmentation_mode

    def explain(self,
                model: nn.Module,
                input_tensor: torch.Tensor,
                class_idx: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """Generate LIME explanations."""
        model.eval()

        if self._segmentation_mode:
            return self._explainSegmentation(model, input_tensor, class_idx, **kwargs)
        else:
            return self._explainClassification(model, input_tensor, **kwargs)

    def _explainSegmentation(self, model, input_tensor, class_idx=None, **kwargs):
        """Complete LIME explanation for knee segmentation."""

        # 1. Determine target class ONCE
        if class_idx is None:
            class_idx = self._findMostCommonKneeClass(model, input_tensor)

        # 2. Create forward function for THIS class
        def forward_func(x):
            return self._getKneeClassProbabilities(model, x)

        lime = Lime(forward_func)

        attr = lime.attribute(
            input_tensor,
            target=class_idx,
            n_samples=self._n_samples,
            perturbations_per_eval=self._batch_size,
            **kwargs
        )

        return attr

    def _explainClassification(self,
                               model: nn.Module,
                               input_tensor: torch.Tensor,
                               **kwargs) -> torch.Tensor:
        with torch.no_grad():
            output = model(input_tensor)
            target = output.argmax(dim=1)

        lime = Lime(model)
        attr = lime.attribute(
            input_tensor,
            target=target,
            n_samples=self._n_samples,
            perturbations_per_eval=self._batch_size,
            **kwargs
        )

        return attr

    def getName(self) -> str:
        return f"LIMEExplainer(n_samples={self._n_samples})"


class SaliencyExplainer(KneeMRIDatasetExplainer):
    """Saliency explanation for medical image segmentation using Captum."""

    def __init__(self,
                 absolute: bool = True):
        """
        Args:
            absolute: If True, return absolute values of gradients
            smooth_grad: If True, use SmoothGrad to reduce noise
            n_samples: Number of samples for SmoothGrad
            stdevs: Standard deviation for noise in SmoothGrad
        """
        self._absolute = absolute

    def explain(self,
                model: nn.Module,
                input_tensor: torch.Tensor,
                target_class: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """Generate saliency explanations for segmentation."""
        model.eval()

        # Determine target class if not provided
        if target_class is None:
            target_class = self._findMostCommonKneeClass(model, input_tensor)

        def forward_func(x):
            return self._getKneeClassProbabilities(model, x, use_gradients=True)

        saliency = Saliency(forward_func)

        attr = saliency.attribute(
            input_tensor,
            target=target_class,
            abs=self._absolute,
            **kwargs
        )

        return attr

    def _explainClassification(self,
                               model: nn.Module,
                               input_tensor: torch.Tensor,
                               target: Optional[torch.Tensor] = None,
                               **kwargs) -> torch.Tensor:
        """Saliency for classification models (fallback)."""
        saliency = Saliency(model)

        if target is None:
            with torch.no_grad():
                output = model(input_tensor)
                target = output.argmax(dim=1)

        attr = saliency.attribute(input_tensor, target=target, abs=self._absolute **kwargs)

        return attr

    def getName(self) -> str:
        return f"SaliencyExplainer(absolute={self._absolute})"


class GradCAMExplainer(KneeMRIDatasetExplainer):

    def __init__(self,
                 layer_name: Optional[str] = None):
        self._layer_name = layer_name

    def explain(self,
                model: nn.Module,
                input_tensor: torch.Tensor,
                target_class: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        model.eval()

        if target_class is None:
            target_class = self._findMostCommonKneeClass(model, input_tensor)

        target_layer = self._findConvLayer(model, self._layer_name)
        if target_layer is None:
            raise ValueError("No conv layer for Grad-CAM")

        def forward_func(x):
            return self._getKneeClassProbabilities(model, x, use_gradients=True)

        grad_cam = LayerGradCam(forward_func, target_layer[1])

        attr = grad_cam.attribute(
            input_tensor,
            target=target_class,
            **kwargs
        )

        attr = LayerAttribution.interpolate(attr, input_tensor.shape[2:])
        attr = F.relu(attr)

        return attr.detach()

    def _findConvLayer(self, model, layer_name=None):
        conv_layers = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                conv_layers.append((name, module))

        if not conv_layers:
            return None

        if layer_name:
            for name, module in conv_layers:
                if name == layer_name:
                    return (name, module)

        return conv_layers[-1]

    def getName(self) -> str:
        layer_info = f"layer={self._layer_name}" if self._layer_name else "last-conv"
        return f"GradCAMExplainer({layer_info})"


class XAIManager:
    """Main manager for XAI operations."""

    def __init__(self, model: nn.Module):
        """Initialize XAI Manager."""
        self._model = model
        self._extraction_command = PyTorchHookExtractionCommand()
        self._explainer: Optional[XAIExplanationMethod] = None

    def setExplainer(self, explainer: XAIExplanationMethod):
        """Set the explanation method to use."""
        self._explainer = explainer

    def explain(self,
                input_tensor: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Generate explanations for input."""
        if self._explainer is None:
            raise ValueError("No explainer set. Use setExplainer() first.")

        return self._explainer.explain(self._model, input_tensor, target, **kwargs)

    def gatherActivations(self,
                          dataloader: DataLoader,
                          layer_names: LayerNameList) -> ActivationDict:
        return self._extraction_command.extract(self._model, dataloader, layer_names)

    def saveActivations(self,
                        activations: ActivationDict,
                        save_path: Union[str, Path],
                        metadata: Optional[Dict] = None):
        if metadata is None:
            metadata = {}

        metadata.update({
            'model_type': type(self._model).__name__,
            'timestamp': datetime.now().isoformat()
        })

        ActivationStorage.saveActivations(activations, save_path, metadata)

    def analyzeLayersSensitivity(self,
                                 dataloader: DataLoader,
                                 modification_command: InputModificationCommand,
                                 layer_names: Optional[LayerNameList] = None) -> SensitivityAnalysisResult:
        """Analyze sensitivity of layers to input modifications."""
        if layer_names is None:
            layer_names = [name for name, _ in self._model.named_modules() if name]

        original_activations = self.gatherActivations(dataloader, layer_names)
        modified_activations = self._getModifiedActivations(dataloader, modification_command, layer_names)

        return self._calculateSensitivityResults(
            original_activations, modified_activations,
            layer_names, modification_command.getDescription()
        )

    def _getModifiedActivations(self,
                                dataloader: DataLoader,
                                modification_command: InputModificationCommand,
                                layer_names: LayerNameList) -> ActivationDict:
        """Get activations for modified inputs."""
        activations_list = []

        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                modified_batch = self._modifyBatch(batch, modification_command)
                modified_dataloader = [modified_batch]

                batch_activations = self._extraction_command.extract(
                    self._model, modified_dataloader, layer_names
                )
                activations_list.append(batch_activations)

        return ActivationConcatenator.concatenateActivations(activations_list)

    def _modifyBatch(self, batch: BatchType, modification_command: InputModificationCommand) -> BatchType:
        """Apply modification to a batch."""
        if isinstance(batch, (list, tuple)):
            input_tensor = batch[0]
            rest = batch[1:] if len(batch) > 1 else ()
            modified_input = modification_command.modify(input_tensor)
            return (modified_input,) + rest
        return modification_command.modify(batch)

    def _calculateSensitivityResults(self,
                                     original: ActivationDict,
                                     modified: ActivationDict,
                                     all_layer_names: LayerNameList,
                                     modification_description: str) -> SensitivityAnalysisResult:
        sensitivity_scores = []
        valid_layer_names = []

        for layer_name in all_layer_names:
            if layer_name in original and layer_name in modified:
                orig = original[layer_name]
                mod = modified[layer_name]

                if orig.shape == mod.shape:
                    mean_diff = torch.abs(orig - mod).mean().item()
                    sensitivity_scores.append(mean_diff)
                    valid_layer_names.append(layer_name)
                else:
                    raise ValueError(f"Input and mofified tensors have different dimensions: {orig.shape} {mod.shape}")

        return SensitivityAnalysisResult(
            layer_names=valid_layer_names,
            sensitivity_scores=sensitivity_scores,
            modification_description=modification_description
        )

    def visualizeSensitivity(self,
                             result: SensitivityAnalysisResult,
                             save_path: Optional[Union[str, Path]] = None):
        SensitivityVisualizer.plotSensitivityBarChart(result, save_path)

    def setExtractionCommand(self, command: ActivationExtractionCommand):
        self._extraction_command = command
