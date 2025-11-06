from abc import ABC, abstractmethod
from typing import Dict, List
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


class GANMetrics(Metrics):
    """Metrics tracking for GAN training"""

    def __init__(self):
        self._history = {
            'train': [],
            'val': []
        }
        self._batch_data = {
            'train': {
                'generator_losses': [],
                'discriminator_losses': [],
                'discriminator_real_acc': [],
                'discriminator_fake_acc': []
            },
            'val': {
                'generator_losses': [],
                'discriminator_losses': [],
                'discriminator_real_acc': [],
                'discriminator_fake_acc': []
            }
        }
        self._current_phase = 'train'
        self._current_epoch_metrics = {
            'train': None,
            'val': None
        }

    def setPhase(self, phase: str) -> None:
        """Set current phase (train/val)"""
        if phase not in ['train', 'val']:
            raise ValueError(f"Phase must be 'train' or 'val', got {phase}")
        self._current_phase = phase

    def accumulate(self, loss_dict: Dict[str, float]) -> None:
        """
        Accumulate metrics during batch processing.

        Args:
            loss_dict: Dictionary containing 'generator_loss', 'discriminator_loss',
                      'discriminator_real_acc', 'discriminator_fake_acc'
        """
        phase_data = self._batch_data[self._current_phase]
        phase_data['generator_losses'].append(loss_dict.get('generator_loss', 0.0))
        phase_data['discriminator_losses'].append(loss_dict.get('discriminator_loss', 0.0))
        phase_data['discriminator_real_acc'].append(loss_dict.get('discriminator_real_acc', 0.0))
        phase_data['discriminator_fake_acc'].append(loss_dict.get('discriminator_fake_acc', 0.0))

    def finalizeEpoch(self, epoch: int) -> None:
        """
        Calculate and store final metrics for both train and val phases.

        Args:
            epoch: Current epoch number
        """
        for phase in ['train', 'val']:
            phase_data = self._batch_data[phase]

            if not phase_data['generator_losses']:
                self._current_epoch_metrics[phase] = None
                continue

            avg_gen_loss = np.mean(phase_data['generator_losses'])
            avg_disc_loss = np.mean(phase_data['discriminator_losses'])
            avg_real_acc = np.mean(phase_data['discriminator_real_acc'])
            avg_fake_acc = np.mean(phase_data['discriminator_fake_acc'])

            metrics_dict = {
                'epoch': epoch,
                'generator_loss': avg_gen_loss,
                'discriminator_loss': avg_disc_loss,
                'discriminator_real_acc': avg_real_acc,
                'discriminator_fake_acc': avg_fake_acc
            }

            self._history[phase].append(metrics_dict)
            self._current_epoch_metrics[phase] = metrics_dict

            self._batch_data[phase] = {
                'generator_losses': [],
                'discriminator_losses': [],
                'discriminator_real_acc': [],
                'discriminator_fake_acc': []
            }

    def display(self, epoch: int) -> None:
        """
        Display aggregated metrics for both train and val phases.

        Args:
            epoch: Current epoch number
        """
        for phase in ['train', 'val']:
            metrics_dict = self._current_epoch_metrics[phase]

            if metrics_dict is None:
                print(f"Epoch {epoch} [{phase}]: No data")
                continue

            avg_gen_loss = metrics_dict['generator_loss']
            avg_disc_loss = metrics_dict['discriminator_loss']
            avg_real_acc = metrics_dict['discriminator_real_acc']
            avg_fake_acc = metrics_dict['discriminator_fake_acc']

            phase_label = "Train" if phase == 'train' else "Val  "
            print(
                f"Epoch {epoch} [{phase_label}]: "
                f"G loss: {avg_gen_loss:.4f} | "
                f"D loss: {avg_disc_loss:.4f} | "
                f"D real acc: {avg_real_acc:.2f} | "
                f"D fake acc: {avg_fake_acc:.2f}"
            )

            if phase == 'val':
                if avg_real_acc < 0.5 and avg_fake_acc < 0.5:
                    print("  ⚠️  WARNING: Discriminator performing poorly on both real and fake!")
                elif avg_real_acc > 0.9 and avg_fake_acc > 0.9:
                    print("  ⚠️  WARNING: Discriminator too strong - generator may not learn!")

        if self._history['val']:
            best_gen_loss = min(entry['generator_loss'] for entry in self._history['val'])
            print(f"  Best val generator loss: {best_gen_loss:.4f}")

    def getMetric(self, epoch: int, phase: str, metric_name: str) -> float:
        """
        Get a specific metric value for early stopping.

        Args:
            epoch: Epoch number
            phase: 'train' or 'val'
            metric_name: Name of metric (e.g., 'generator_loss', 'discriminator_loss')

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
            model_type: Type of model (e.g., 'gan')
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

    def getHistoryLists(self) -> Dict[str, List[float]]:
        """
        Get metrics history as lists (for backward compatibility).

        Returns:
            Dictionary containing lists of metrics
        """
        result = {
            'generator_train_losses': [],
            'discriminator_train_losses': [],
            'discriminator_train_real_acc': [],
            'discriminator_train_fake_acc': [],
            'generator_val_losses': [],
            'discriminator_val_losses': [],
            'discriminator_val_real_acc': [],
            'discriminator_val_fake_acc': []
        }

        for entry in self._history['train']:
            result['generator_train_losses'].append(entry['generator_loss'])
            result['discriminator_train_losses'].append(entry['discriminator_loss'])
            result['discriminator_train_real_acc'].append(entry['discriminator_real_acc'])
            result['discriminator_train_fake_acc'].append(entry['discriminator_fake_acc'])

        for entry in self._history['val']:
            result['generator_val_losses'].append(entry['generator_loss'])
            result['discriminator_val_losses'].append(entry['discriminator_loss'])
            result['discriminator_val_real_acc'].append(entry['discriminator_real_acc'])
            result['discriminator_val_fake_acc'].append(entry['discriminator_fake_acc'])

        return result


class VAEMetrics(Metrics):
    """Metrics tracking for VAE training"""

    def __init__(self):
        self._history = {
            'train': [],
            'val': []
        }
        self._batch_data = {
            'train': {
                'total_losses': [],
                'reconstruction_losses': [],
                'kl_divergences': []
            },
            'val': {
                'total_losses': [],
                'reconstruction_losses': [],
                'kl_divergences': []
            }
        }
        self._current_phase = 'train'
        self._current_epoch_metrics = {
            'train': None,
            'val': None
        }

    def setPhase(self, phase: str) -> None:
        """Set current phase (train/val)"""
        if phase not in ['train', 'val']:
            raise ValueError(f"Phase must be 'train' or 'val', got {phase}")
        self._current_phase = phase

    def accumulate(self, loss_dict: Dict[str, float]) -> None:
        """
        Accumulate metrics during batch processing.

        Args:
            loss_dict: Dictionary containing 'total_loss', 'reconstruction_loss', 'kl_divergence'
        """
        phase_data = self._batch_data[self._current_phase]
        phase_data['total_losses'].append(loss_dict.get('total_loss', 0.0))
        phase_data['reconstruction_losses'].append(loss_dict.get('reconstruction_loss', 0.0))
        phase_data['kl_divergences'].append(loss_dict.get('kl_divergence', 0.0))

    def finalizeEpoch(self, epoch: int) -> None:
        """
        Calculate and store final metrics for both train and val phases.

        Args:
            epoch: Current epoch number
        """
        for phase in ['train', 'val']:
            phase_data = self._batch_data[phase]

            if not phase_data['total_losses']:
                self._current_epoch_metrics[phase] = None
                continue

            avg_total_loss = np.mean(phase_data['total_losses'])
            avg_recon_loss = np.mean(phase_data['reconstruction_losses'])
            avg_kl_div = np.mean(phase_data['kl_divergences'])

            metrics_dict = {
                'epoch': epoch,
                'total_loss': avg_total_loss,
                'reconstruction_loss': avg_recon_loss,
                'kl_divergence': avg_kl_div
            }

            self._history[phase].append(metrics_dict)
            self._current_epoch_metrics[phase] = metrics_dict

            self._batch_data[phase] = {
                'total_losses': [],
                'reconstruction_losses': [],
                'kl_divergences': []
            }

    def display(self, epoch: int) -> None:
        """
        Display aggregated metrics for both train and val phases.

        Args:
            epoch: Current epoch number
        """
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
            print(f"Best val total loss: {best_total_loss:.6f}")

    def getMetric(self, epoch: int, phase: str, metric_name: str) -> float:
        """
        Get a specific metric value for early stopping.

        Args:
            epoch: Epoch number
            phase: 'train' or 'val'
            metric_name: Name of metric (e.g., 'total_loss', 'reconstruction_loss', 'kl_divergence')

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
            model_type: Type of model (e.g., 'vae')
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

    def getHistoryLists(self) -> Dict[str, List[float]]:
        """
        Get metrics history as lists (for backward compatibility or plotting).

        Returns:
            Dictionary containing lists of metrics
        """
        result = {
            'train_total_losses': [],
            'train_reconstruction_losses': [],
            'train_kl_divergences': [],
            'val_total_losses': [],
            'val_reconstruction_losses': [],
            'val_kl_divergences': []
        }

        for entry in self._history['train']:
            result['train_total_losses'].append(entry['total_loss'])
            result['train_reconstruction_losses'].append(entry['reconstruction_loss'])
            result['train_kl_divergences'].append(entry['kl_divergence'])

        for entry in self._history['val']:
            result['val_total_losses'].append(entry['total_loss'])
            result['val_reconstruction_losses'].append(entry['reconstruction_loss'])
            result['val_kl_divergences'].append(entry['kl_divergence'])

        return result


class GeneratorMetrics(Metrics):
    """Metrics tracking for Generator training"""

    def __init__(self):
        self._history = {
            'train': [],
            'val': []
        }
        self._batch_data = {
            'train': {
                'total_losses': [],
                'reconstruction_losses': [],
                'perceptual_losses': []
            },
            'val': {
                'total_losses': [],
                'reconstruction_losses': [],
                'perceptual_losses': []
            }
        }
        self._current_phase = 'train'
        self._current_epoch_metrics = {
            'train': None,
            'val': None
        }

    def setPhase(self, phase: str) -> None:
        """Set current phase (train/val)"""
        if phase not in ['train', 'val']:
            raise ValueError(f"Phase must be 'train' or 'val', got {phase}")
        self._current_phase = phase

    def accumulate(self, loss_dict: Dict[str, float]) -> None:
        """
        Accumulate metrics during batch processing.

        Args:
            loss_dict: Dictionary containing 'total_loss', 'reconstruction_loss', 'perceptual_loss'
        """
        phase_data = self._batch_data[self._current_phase]
        phase_data['total_losses'].append(loss_dict.get('total_loss', 0.0))
        phase_data['reconstruction_losses'].append(loss_dict.get('reconstruction_loss', 0.0))
        phase_data['perceptual_losses'].append(loss_dict.get('perceptual_loss', 0.0))

    def finalizeEpoch(self, epoch: int) -> None:
        """
        Calculate and store final metrics for both train and val phases.

        Args:
            epoch: Current epoch number
        """
        for phase in ['train', 'val']:
            phase_data = self._batch_data[phase]

            if not phase_data['total_losses']:
                self._current_epoch_metrics[phase] = None
                continue

            avg_total_loss = np.mean(phase_data['total_losses'])
            avg_recon_loss = np.mean(phase_data['reconstruction_losses'])
            avg_percept_loss = np.mean(phase_data['perceptual_losses'])

            metrics_dict = {
                'epoch': epoch,
                'total_loss': avg_total_loss,
                'reconstruction_loss': avg_recon_loss,
                'perceptual_loss': avg_percept_loss
            }

            self._history[phase].append(metrics_dict)
            self._current_epoch_metrics[phase] = metrics_dict

            self._batch_data[phase] = {
                'total_losses': [],
                'reconstruction_losses': [],
                'perceptual_losses': []
            }

    def display(self, epoch: int) -> None:
        """
        Display aggregated metrics for both train and val phases.

        Args:
            epoch: Current epoch number
        """
        for phase in ['train', 'val']:
            metrics_dict = self._current_epoch_metrics[phase]

            if metrics_dict is None:
                continue

            phase_label = "Train" if phase == 'train' else "Val  "

            if metrics_dict['perceptual_loss'] > 0:
                print(
                    f"Epoch {epoch} [{phase_label}]: "
                    f"Total loss: {metrics_dict['total_loss']:.6f} | "
                    f"Recon loss: {metrics_dict['reconstruction_loss']:.6f} | "
                    f"Percept loss: {metrics_dict['perceptual_loss']:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch} [{phase_label}]: "
                    f"Loss: {metrics_dict['total_loss']:.6f}"
                )

        if self._history['val']:
            best_total_loss = min(entry['total_loss'] for entry in self._history['val'])
            print(f"  Best val loss: {best_total_loss:.6f}")

    def getMetric(self, epoch: int, phase: str, metric_name: str) -> float:
        """
        Get a specific metric value for early stopping.

        Args:
            epoch: Epoch number
            phase: 'train' or 'val'
            metric_name: Name of metric (e.g., 'total_loss', 'reconstruction_loss')

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
            model_type: Type of model (e.g., 'generator')
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

    def getHistoryLists(self) -> Dict[str, List[float]]:
        """
        Get metrics history as lists (for backward compatibility).

        Returns:
            Dictionary containing lists of metrics
        """
        result = {
            'train_losses': [],
            'train_reconstruction_losses': [],
            'train_perceptual_losses': [],
            'val_losses': [],
            'val_reconstruction_losses': [],
            'val_perceptual_losses': []
        }

        for entry in self._history['train']:
            result['train_losses'].append(entry['total_loss'])
            result['train_reconstruction_losses'].append(entry['reconstruction_loss'])
            result['train_perceptual_losses'].append(entry['perceptual_loss'])

        for entry in self._history['val']:
            result['val_losses'].append(entry['total_loss'])
            result['val_reconstruction_losses'].append(entry['reconstruction_loss'])
            result['val_perceptual_losses'].append(entry['perceptual_loss'])

        return result
