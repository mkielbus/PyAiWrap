from typing import Dict, Tuple
import torch.nn as nn
import torch
from .visualize import visualizeReconstruction
import os
import matplotlib.pyplot as plt
import numpy as np


class GeneratorControlFunc:
    """
    Callable class for visualizing generator reconstruction during training.
    """

    def __init__(self, target_channel: str = "RGB", input_channel: str = "RGB"):
        """
        Initialize the control function with target channel.

        Args:
            target_channel: Target channel for single-channel models ("R", "G", "B", or "RGB")
        """
        self.target_channel = target_channel
        self.input_channel = input_channel

    def __call__(
        self,
        models: Dict[str, nn.Module],
        train_batch: Tuple,
        val_batch: Tuple,
        epoch: int,
        diagrams_path: str,
        hyperparams_id: str,
        model_type: str,
        launch_number: int
    ) -> None:
        """
        Control function for visualizing generator reconstruction during training.

        Args:
            models: Dictionary containing the generator model
            train_batch: Batch from training data (modified_images, original_images)
            val_batch: Batch from validation data (modified_images, original_images)
            epoch: Current epoch number
            diagrams_path: Path to save visualizations
            hyperparams_id: Hyperparameter configuration ID
            model_type: Type of model (for naming files)
            launch_number: Launch number for this training run
        """
        generator = models['generator']
        generator.eval()

        train_modified, train_original = train_batch[0], train_batch[1]
        val_modified, val_original = val_batch[0], val_batch[1]

        with torch.no_grad():
            train_reconstructed = generator(train_modified)
            val_reconstructed = generator(val_modified)

        visualizeReconstruction(
            originalImages=train_original,
            modifiedImages=train_modified,
            reconstructedImages=train_reconstructed,
            epoch=epoch,
            savePath=diagrams_path,
            modelType=f"train_{model_type}",
            launchNumber=launch_number,
            hyperparamsId=hyperparams_id,
            numImages=8,
            targetChannel=self.target_channel,
            inputChannel=self.input_channel
        )

        visualizeReconstruction(
            originalImages=val_original,
            modifiedImages=val_modified,
            reconstructedImages=val_reconstructed,
            epoch=epoch,
            savePath=diagrams_path,
            modelType=f"val_{model_type}",
            launchNumber=launch_number,
            hyperparamsId=hyperparams_id,
            numImages=8,
            targetChannel=self.target_channel,
            inputChannel=self.input_channel
        )


class VAEControlFunc:
    """
    Callable class for visualizing VAE reconstruction during training.
    """

    def __init__(self, target_channel: str = "RGB"):
        """
        Initialize the control function with target channel.

        Args:
            target_channel: Target channel for single-channel models ("R", "G", "B", or "RGB")
        """
        self.target_channel = target_channel

    def __call__(
        self,
        models: Dict[str, nn.Module],
        train_batch: Tuple,
        val_batch: Tuple,
        epoch: int,
        diagrams_path: str,
        hyperparams_id: str,
        model_type: str,
        launch_number: int
    ) -> None:
        """
        Control function for visualizing VAE reconstruction during training.

        Args:
            models: Dictionary containing the VAE model
            train_batch: Batch from training data (modified_images, original_images)
            val_batch: Batch from validation data (modified_images, original_images)
            epoch: Current epoch number
            diagrams_path: Path to save visualizations
            hyperparams_id: Hyperparameter configuration ID
            model_type: Type of model (for naming files)
            launch_number: Launch number for this training run
        """
        vae = models['vae']
        vae.eval()

        train_modified, train_original = train_batch[0], train_batch[1]
        val_modified, val_original = val_batch[0], val_batch[1]

        with torch.no_grad():
            train_reconstructed, train_mu, train_logvar = vae(train_modified)
            val_reconstructed, val_mu, val_logvar = vae(val_modified)

        visualizeReconstruction(
            original_images=train_original,
            modified_images=train_modified,
            reconstructed_images=train_reconstructed,
            epoch=epoch,
            save_path=diagrams_path,
            model_type=f"{model_type}_train",
            hyperparams_id=hyperparams_id,
            launch_number=launch_number,
            num_images=8,
            target_channel=self.target_channel
        )

        visualizeReconstruction(
            original_images=val_original,
            modified_images=val_modified,
            reconstructed_images=val_reconstructed,
            epoch=epoch,
            save_path=diagrams_path,
            model_type=f"{model_type}_val",
            hyperparams_id=hyperparams_id,
            launch_number=launch_number,
            num_images=8,
            target_channel=self.target_channel
        )


class GANControlFunc:
    """
    Callable class for visualizing GAN generation during training.
    """

    def __init__(self, target_channel: str = "RGB"):
        """
        Initialize the control function with target channel.

        Args:
            target_channel: Target channel for single-channel models ("R", "G", "B", or "RGB")
        """
        self.target_channel = target_channel

    def __call__(
        self,
        models: Dict[str, nn.Module],
        train_batch: Tuple,
        val_batch: Tuple,
        epoch: int,
        diagrams_path: str,
        hyperparams_id: str,
        model_type: str,
        launch_number: int
    ) -> None:
        """
        Control function for visualizing GAN generation during training.

        Args:
            models: Dictionary containing generator and discriminator models
            train_batch: Batch from training data
            val_batch: Batch from validation data
            epoch: Current epoch number
            diagrams_path: Path to save visualizations
            hyperparams_id: Hyperparameter configuration ID
            model_type: Type of model (for naming files)
            launch_number: Launch number for this training run
        """
        generator = models['generator']
        generator.eval()

        train_modified, train_original = train_batch[1], train_batch[1]
        val_modified, val_original = val_batch[0], val_batch[1]

        with torch.no_grad():
            train_generated = generator(train_modified)
            val_generated = generator(val_modified)

        visualizeReconstruction(
            original_images=train_original,
            modified_images=train_modified,
            reconstructed_images=train_generated,
            epoch=epoch,
            save_path=diagrams_path,
            model_type=f"{model_type}_train",
            hyperparams_id=hyperparams_id,
            launch_number=launch_number,
            num_images=8,
            target_channel=self.target_channel
        )

        visualizeReconstruction(
            original_images=val_original,
            modified_images=val_modified,
            reconstructed_images=val_generated,
            epoch=epoch,
            save_path=diagrams_path,
            model_type=f"{model_type}_val",
            hyperparams_id=hyperparams_id,
            launch_number=launch_number,
            num_images=8,
            target_channel=self.target_channel
        )


class SegmentationControlFunc:
    """
    Callable class for visualizing segmentation results during training.
    """

    def __init__(self, num_classes: int = 4):
        """
        Initialize the segmentation control function.

        Args:
            num_classes: Number of segmentation classes
        """
        self.num_classes = num_classes
        # Subtle colors for ACL classes only (background remains unchanged)
        self.class_colors = {
            1: [0.2, 0.8, 0.2],  # Healthy - subtle green
            2: [0.9, 0.9, 0.2],  # Partial injury - subtle yellow
            3: [0.9, 0.3, 0.3]   # Complete rupture - subtle red
        }

    def __call__(
        self,
        models: Dict[str, nn.Module],
        train_batch: Tuple,
        val_batch: Tuple,
        epoch: int,
        diagrams_path: str,
        hyperparams_id: str,
        model_type: str,
        launch_number: int
    ) -> None:
        """
        Control function for visualizing segmentation during training.

        Args:
            models: Dictionary containing the segmentation model
            train_batch: Batch from training data (volumes, seg_masks)
            val_batch: Batch from validation data (volumes, seg_masks)
            epoch: Current epoch number
            diagrams_path: Path to save visualizations
            hyperparams_id: Hyperparameter configuration ID
            model_type: Type of model (for naming files)
            launch_number: Launch number for this training run
        """
        seg_model = models['segformer']
        seg_model.eval()

        train_volumes, train_masks = train_batch[0], train_batch[1]
        val_volumes, val_masks = val_batch[0], val_batch[1]

        with torch.no_grad():
            train_preds = seg_model(train_volumes)
            val_preds = seg_model(val_volumes)

        self.visualizeSegmentation(
            volumes=train_volumes,
            true_masks=train_masks,
            pred_logits=train_preds,
            epoch=epoch,
            save_path=diagrams_path,
            phase="train",
            hyperparams_id=hyperparams_id,
            model_type=model_type,
            launch_number=launch_number
        )

        self.visualizeSegmentation(
            volumes=val_volumes,
            true_masks=val_masks,
            pred_logits=val_preds,
            epoch=epoch,
            save_path=diagrams_path,
            phase="val",
            hyperparams_id=hyperparams_id,
            model_type=model_type,
            launch_number=launch_number
        )

    def visualizeSegmentation(
        self,
        volumes: torch.Tensor,
        true_masks: torch.Tensor,
        pred_logits: torch.Tensor,
        epoch: int,
        save_path: str,
        phase: str,
        hyperparams_id: str,
        model_type: str,
        launch_number: int
    ) -> None:
        """
        Visualize segmentation results with ground truth and predictions.

        Args:
            volumes: Input volumes [B, 1, D, H, W]
            true_masks: Ground truth masks [B, 1, D, H, W]
            pred_logits: Model predictions [B, C, D, H, W]
            epoch: Current epoch
            save_path: Path to save images
            phase: 'train' or 'val'
            hyperparams_id: Hyperparameter ID
            model_type: Model type
            launch_number: Launch number
        """
        os.makedirs(save_path, exist_ok=True)

        # Get center slices and convert to numpy
        batch_size = min(4, volumes.shape[0])
        volumes_np = volumes.cpu().numpy()
        true_masks_np = true_masks.cpu().numpy()

        # Remove channel dimension from true masks for visualization
        if true_masks_np.ndim == 5:  # [B, 1, D, H, W]
            true_masks_np = true_masks_np[:, 0]  # [B, D, H, W]

        pred_masks_np = torch.softmax(pred_logits, dim=1).argmax(dim=1).cpu().numpy()

        fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
        if batch_size == 1:
            axes = axes.reshape(2, 1)

        for i in range(batch_size):
            depth = volumes_np[i, 0].shape[0]
            center_slice = depth // 2

            input_slice = volumes_np[i, 0, center_slice]
            true_mask_slice = true_masks_np[i, center_slice]

            pred_mask_slice = pred_masks_np[i, center_slice]

            true_colored = self.maskToColor(input_slice, true_mask_slice)
            pred_colored = self.maskToColor(input_slice, pred_mask_slice)

            axes[0, i].imshow(true_colored)
            axes[0, i].set_title(f'GT Sample {i+1}')
            axes[0, i].axis('off')

            axes[1, i].imshow(pred_colored)
            axes[1, i].set_title(f'Pred Sample {i+1}')
            axes[1, i].axis('off')

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', label='Background'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.2, 0.8, 0.2], label='Healthy'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.9, 0.2], label='Partially injured'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.3, 0.3], label='Completely ruptured')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   bbox_to_anchor=(0.5, -0.05))

        plt.suptitle(f'Epoch {epoch} - {phase.upper()} - Segmentation Results')
        plt.tight_layout()

        filename = f"{model_type}_{phase}_segmentation_epoch_{epoch}_hyperparams_{hyperparams_id}_{launch_number}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()

    def maskToColor(self, input_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
        """
        Convert mask to colored overlay on input image.

        Args:
            input_slice: Grayscale input slice [H, W]
            mask_slice: Segmentation mask slice [H, W]

        Returns:
            Colored RGB image [H, W, 3]
        """
        input_normalized = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8)

        colored = np.stack([input_normalized] * 3, axis=-1)

        for class_id, color in self.class_colors.items():
            mask = mask_slice == class_id
            if mask.any():
                for channel in range(3):
                    colored_channel = colored[..., channel]
                    colored_channel[mask] = color[channel] * 0.3 + colored_channel[mask] * 0.7
                    colored[..., channel] = colored_channel

        return np.clip(colored, 0, 1)
