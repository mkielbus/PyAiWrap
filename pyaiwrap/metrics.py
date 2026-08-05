from abc import ABC, abstractmethod
from typing import Any, Dict, List
import copy
import json
import os
import numpy as np


class Metrics(ABC):
    """Abstract base class for metrics tracking"""

    @abstractmethod
    def finalizeEpoch(self, epoch: int) -> None:
        """Finalize and calculate metrics for the current epoch"""
        pass

    @abstractmethod
    def display(self, epoch: int) -> None:
        """Display metrics for a given epoch"""
        pass

    @abstractmethod
    def save(self, path: str, hyperparams_id: str, model_type: str, launch_number: int) -> None:
        """Save metrics to file"""
        pass

    @abstractmethod
    def getMetric(self, epoch: int, phase: str, metric_name: str) -> float:
        """Get a specific metric value"""
        pass

    @abstractmethod
    def setPhase(self, phase: str) -> None:
        """Set current phase (train/val)"""
        pass


class BaseMetrics(Metrics):
    """Base implementation of common metrics functionality"""

    def __init__(self, metric_keys: List[str]):
        """
        Initialize base metrics.

        Args:
            metric_keys: List of metric names to track
        """
        self.metric_keys = metric_keys
        self._history = {'train': [], 'val': []}
        self._batch_data = {
            'train': {key: [] for key in metric_keys},
            'val': {key: [] for key in metric_keys}
        }
        self._current_phase = 'train'
        self._current_epoch_metrics = {'train': None, 'val': None}

    def setPhase(self, phase: str) -> None:
        """Set current phase (train/val)"""
        if phase not in ['train', 'val']:
            raise ValueError(f"Phase must be 'train' or 'val', got {phase}")
        self._current_phase = phase

    def accumulate(self, loss_dict: Dict[str, float]) -> None:
        """
        Accumulate metrics during batch processing.

        Args:
            loss_dict: Dictionary containing metric values
        """
        phase_data = self._batch_data[self._current_phase]
        for key in self.metric_keys:
            phase_data[key].append(loss_dict.get(key, 0.0))

    def finalizeEpoch(self, epoch: int) -> None:
        """
        Calculate and store final metrics for both train and val phases.

        Args:
            epoch: Current epoch number
        """
        for phase in ['train', 'val']:
            phase_data = self._batch_data[phase]

            first_key = self.metric_keys[0]
            if not phase_data[first_key]:
                self._current_epoch_metrics[phase] = None
                continue

            metrics_dict = {'epoch': epoch}
            for key in self.metric_keys:
                # plain float, not np.float64: checkpoints holding these values must
                # stay loadable under torch.load(weights_only=True) (torch >= 2.6 default)
                metrics_dict[key] = float(np.mean(phase_data[key]))

            self._history[phase].append(metrics_dict)
            self._current_epoch_metrics[phase] = metrics_dict

            self._batch_data[phase] = {key: [] for key in self.metric_keys}

    def getMetric(self, epoch: int, phase: str, metric_name: str) -> float:
        """
        Get a specific metric value for early stopping.

        Args:
            epoch: Epoch number
            phase: 'train' or 'val'
            metric_name: Name of metric

        Returns:
            Metric value or inf if not found
        """
        for entry in self._history[phase]:
            if entry['epoch'] == epoch:
                return entry.get(metric_name, float('inf'))
        return float('inf')

    def save(self, path: str, hyperparams_id: str, model_type: str, launch_number: int) -> None:
        """
        Save metrics to JSON file.

        Args:
            path: Directory path to save metrics
            hyperparams_id: Hyperparameter configuration ID
            model_type: Type of model
            launch_number: Launch number for this training run
        """
        os.makedirs(path, exist_ok=True)

        metrics_data = {
            "hyperparams_id": hyperparams_id,
            "model_type": model_type,
            "launch_number": launch_number,
            "train": self._history['train'],
            "val": self._history['val']
        }

        filename = f"{model_type}_metrics_hyperparams_{hyperparams_id}_{launch_number}.json"
        filepath = os.path.join(path, filename)

        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)

    def getState(self) -> Dict[str, Any]:
        """Serializable state for checkpointing (the finalized per-epoch history)"""
        return {'history': copy.deepcopy(self._history)}

    def setState(self, state: Dict[str, Any]) -> None:
        """Restore state produced by getState, e.g. when resuming from a checkpoint"""
        self._history = copy.deepcopy(state['history'])

    def getHistoryLists(self) -> Dict[str, List[float]]:
        """
        Get metrics history as lists.

        Returns:
            Dictionary containing lists of metrics
        """
        result = {}

        for phase in ['train', 'val']:
            for key in self.metric_keys:
                result_key = f"{phase}_{key}"
                # .get, not [key]: a run resumed from a checkpoint written before a metric was
                # added has epochs that predate it, and those epochs must not break the export.
                result[result_key] = [
                    entry.get(key, float('nan')) for entry in self._history[phase]
                ]

        return result

    def _formatMetricValue(self, value: float, precision: int = 6) -> str:
        """Format metric value for display"""
        if abs(value) < 0.01:
            return f"{value:.{precision}f}"
        return f"{value:.4f}"

    @abstractmethod
    def display(self, epoch: int) -> None:
        """Display metrics - must be implemented by subclass"""
        pass


class GANMetrics(BaseMetrics):
    """Metrics tracking for GAN training"""

    def __init__(self):
        metric_keys = [
            'generator_loss',
            'discriminator_loss',
            'discriminator_real_acc',
            'discriminator_fake_acc'
        ]
        super().__init__(metric_keys)

    def display(self, epoch: int) -> None:
        """Display GAN-specific metrics"""
        print()
        for phase in ['train', 'val']:
            metrics_dict = self._current_epoch_metrics[phase]

            if metrics_dict is None:
                continue

            phase_label = "Train" if phase == 'train' else "Val  "
            print(
                f"Epoch {epoch} [{phase_label}]: "
                f"G loss: {metrics_dict['generator_loss']:.4f} | "
                f"D loss: {metrics_dict['discriminator_loss']:.4f} | "
                f"D real acc: {metrics_dict['discriminator_real_acc']:.2f} | "
                f"D fake acc: {metrics_dict['discriminator_fake_acc']:.2f}"
            )

            if phase == 'val':
                real_acc = metrics_dict['discriminator_real_acc']
                fake_acc = metrics_dict['discriminator_fake_acc']

                if real_acc < 0.5 and fake_acc < 0.5:
                    print("  ⚠️  WARNING: Discriminator performing poorly on both real and fake!")
                elif real_acc > 0.9 and fake_acc > 0.9:
                    print("  ⚠️  WARNING: Discriminator too strong - generator may not learn!")

        if self._history['val']:
            best_gen_loss = min(entry['generator_loss'] for entry in self._history['val'])
            print(f"  Best val generator loss: {best_gen_loss:.4f}")


class VAEMetrics(BaseMetrics):
    """Metrics tracking for VAE training"""

    def __init__(self):
        metric_keys = [
            'total_loss',
            'reconstruction_loss',
            'kl_divergence'
        ]
        super().__init__(metric_keys)

    def display(self, epoch: int) -> None:
        """Display VAE-specific metrics"""
        print()
        for phase in ['train', 'val']:
            metrics_dict = self._current_epoch_metrics[phase]

            if metrics_dict is None:
                continue

            phase_label = "Train" if phase == 'train' else "Val  "
            print(
                f"Epoch {epoch} [{phase_label}]: "
                f"Total loss: {metrics_dict['total_loss']:.6f} | "
                f"Recon loss: {metrics_dict['reconstruction_loss']:.6f} | "
                f"KL div: {metrics_dict['kl_divergence']:.6f}"
            )

        if self._history['val']:
            best_total_loss = min(entry['total_loss'] for entry in self._history['val'])
            print(f"  Best val total loss: {best_total_loss:.6f}")


class GeneratorColorizationMetrics(BaseMetrics):
    """Metrics tracking for Generator Colorization training with colorfulness metric"""

    def __init__(self, use_colorfulness: bool = False, use_perceptual_loss: bool = True,
                 track_gradient_norm: bool = False, use_classification: bool = False):
        # *_raw are the unweighted loss terms, tracked alongside the weighted ones so a quality
        # target can be stated in units that do not move when a loss weight is retuned.
        metric_keys = [
            'total_loss',
            'reconstruction_loss',
            'reconstruction_raw'
        ]
        self._use_colorfulness = use_colorfulness
        if self._use_colorfulness:
            metric_keys.extend(['colorfulness_loss',
                                'colorfulness_recon',
                                'colorfulness_original'])
        self._use_perceptual_loss = use_perceptual_loss
        if self._use_perceptual_loss:
            metric_keys.extend(["perceptual_loss", "perceptual_raw"])
        # Pre-clip gradient norm, recorded only when clipping is on because that is the only
        # case where the number decides anything. The validation pass runs without backward and
        # so contributes 0.0, which accumulate() supplies for any key a phase does not emit.
        self._use_classification = use_classification
        if self._use_classification:
            metric_keys.extend(["classification_loss", "classification_raw"])
        self._track_gradient_norm = track_gradient_norm
        if self._track_gradient_norm:
            metric_keys.append("gradient_norm")
        super().__init__(metric_keys)

    def display(self, epoch: int) -> None:
        """Display colorization-specific metrics"""
        print()
        for phase in ['train', 'val']:
            metrics_dict = self._current_epoch_metrics[phase]

            if metrics_dict is None:
                continue

            phase_label = "Train" if phase == 'train' else "Val  "

            loss_parts = [f"Total: {metrics_dict['total_loss']:.6f}"]
            loss_parts.append(f"Recon: {metrics_dict['reconstruction_loss']:.6f}")

            if self._use_perceptual_loss:
                loss_parts.append(f"Percept: {metrics_dict['perceptual_loss']:.6f}")

            if self._use_colorfulness:
                loss_parts.append(f"Color: {metrics_dict['colorfulness_loss']:.6f}")

            if self._use_classification:
                loss_parts.append(f"Class: {metrics_dict['classification_loss']:.6f}")

            print(f"Epoch {epoch} [{phase_label}]: {' | '.join(loss_parts)}")

            raw_parts = [f"L1: {metrics_dict['reconstruction_raw']:.6f}"]
            if self._use_perceptual_loss:
                raw_parts.append(f"LPIPS: {metrics_dict['perceptual_raw']:.6f}")
            if self._track_gradient_norm and phase == 'train':
                raw_parts.append(f"grad norm (pre-clip): "
                                 f"{metrics_dict['gradient_norm']:.3f}")
            print(f"            raw (unweighted) - {', '.join(raw_parts)}")

            if self._use_colorfulness:
                print(f"            Colorfulness - Recon: {metrics_dict['colorfulness_recon']:.2f}, "
                      f"Colorfulness - Original: {metrics_dict['colorfulness_original']:.2f}")

        if self._history['val']:
            best_total_loss = min(entry['total_loss'] for entry in self._history['val'])
            print(f"  Best val loss: {best_total_loss:.6f}")


class SegmentationMetrics(BaseMetrics):
    """Metrics tracking for 3D segmentation training with DiceCE loss"""

    def __init__(self):
        """
        Initialize segmentation metrics for DiceCE loss.
        """
        metric_keys = ['total_loss']
        super().__init__(metric_keys)

    def display(self, epoch: int) -> None:
        """Display DiceCE loss metrics"""
        print()
        for phase in ['train', 'val']:
            metrics_dict = self._current_epoch_metrics[phase]

            if metrics_dict is None:
                continue

            phase_label = "Train" if phase == 'train' else "Val  "

            print(
                f"Epoch {epoch} [{phase_label}]: "
                f"DiceCELoss: {metrics_dict['total_loss']:.4f}"
            )

        if self._history['val']:
            best_loss = min(entry['total_loss'] for entry in self._history['val'])
            current_val_loss = self._current_epoch_metrics['val']['total_loss'] if self._current_epoch_metrics['val'] else float('inf')

            print(f"  Best val DiceCELoss: {best_loss:.4f}")

            if len(self._history['val']) > 1:
                prev_loss = self._history['val'][-2]['total_loss']
                improvement = prev_loss - current_val_loss
                if improvement > 0:
                    print(f"  ↓ Improved by: {improvement:.4f}")
                elif improvement < 0:
                    print(f"  ↑ Worsened by: {abs(improvement):.4f}")
