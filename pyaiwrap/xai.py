from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime

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


class XAIManager:
    """Main manager for XAI operations."""

    def __init__(self, model: nn.Module):
        """Initialize XAI Manager."""
        self._model = model
        self._extraction_command = PyTorchHookExtractionCommand()

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
