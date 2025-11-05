import torch
from typing import Dict, Tuple
import torch.nn as nn
from .visualize import visualizeReconstruction


def generatorControlFunction(
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
        original_images=train_original,
        modified_images=train_modified,
        reconstructed_images=train_reconstructed,
        epoch=epoch,
        save_path=diagrams_path,
        model_type=f"train_{model_type}",
        launch_number=launch_number,
        hyperparams_id=hyperparams_id,
        num_images=8
    )

    visualizeReconstruction(
        original_images=val_original,
        modified_images=val_modified,
        reconstructed_images=val_reconstructed,
        epoch=epoch,
        save_path=diagrams_path,
        model_type=f"val_{model_type}",
        launch_number=launch_number,
        hyperparams_id=hyperparams_id,
        num_images=8
    )
