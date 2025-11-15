from typing import Dict, Tuple
import torch.nn as nn
import torch
from .visualize import visualizeReconstruction


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
