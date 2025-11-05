import torch
import torch.nn as nn
from tqdm import tqdm
import os
from typing import Any, Dict, Callable, Optional, Tuple
from .metrics import Metrics
import json


def loadHyperparameters(json_path: str) -> Dict[str, Any]:
    """
    Load hyperparameters from a JSON file with generic defaults.

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
        "LEARNING_RATE": 0.0001,
        "GAMMA": 0.99,
        "IMAGE_RESIZE": 64,
        "INPUT_CHANNELS": 3,
        "KERNEL_SIZE": 3,
        "EPOCHS": 100,
        "DIAGRAMS_DATA_PATH": "./diagrams_data",
        "WEIGHTS_PATH": "./weights",
        "PATIENCE": 15,
        "DIAGRAMS_PATH": "./diagrams",
        "VISUALIZE_EVERY": 10
    }

    for key, default_value in defaults.items():
        hyperparams.setdefault(key, default_value)

    return hyperparams


def train(
    models: Dict[str, nn.Module],
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    optimizers: Dict[str, torch.optim.Optimizer],
    loss_fn: Callable[[Dict[str, nn.Module], Tuple, Metrics, Optional[float]], Dict[str, Any]],
    metrics: Metrics,
    schedulers: Optional[Dict[str, torch.optim.lr_scheduler._LRScheduler]] = None,
    device: torch.device = torch.device("cuda"),
    num_epochs: int = 100,
    diagrams_data_path: str = "./diagrams_data",
    hyperparams_id: str = "0",
    weights_path: str = "./weights",
    diagrams_path: str = "./diagrams",
    launch_number: int = 0,
    visualize_every_xth_epoch: int = 5,
    max_patience: int = 15,
    model_type: str = "generator",
    gradient_clip: Optional[float] = 1.0,
    control_fn: Optional[Callable] = None,
    early_stopping_metric: str = "loss"
) -> Dict[str, Any]:
    """
    Generic training function for generator models with support for multiple models.

    Args:
        models (Dict[str, nn.Module]): Dictionary of models to train (e.g., {'generator': model1, 'discriminator': model2}).
        train_loader (DataLoader): Training data loader.
        validation_loader (DataLoader): Validation data loader.
        optimizers (Dict[str, Optimizer]): Dictionary of optimizers for each model.
        loss_fn (Callable[[Dict[str, nn.Module], Tuple, Metrics, Optional[float]], Dict[str, Any]]):
                           Loss function that takes (models, batch, metrics, gradient_clip) and returns dict with 'loss' key.
                           Signature: loss_fn(models: Dict[str, nn.Module],
                                            batch: Tuple, 
                                            metrics: Metrics,
                                            gradient_clip: Optional[float]) -> Dict[str, Any]
                           Must return a dict with at least 'loss' key containing the tensor for backpropagation.
                           The function should handle backward() and gradient clipping internally, 
                           and update metrics during each batch.
        metrics (Metrics): Custom metrics object that is updated by loss_fn during batch processing.
                         Should have display(epoch), finalizeEpoch(epoch), setPhase(phase) methods,
                         and optional save() and getMetric() methods.
        schedulers (Optional[Dict[str, _LRScheduler]]): Dictionary of learning rate schedulers for each model.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
        diagrams_data_path (str): Path to save metrics.
        hyperparams_id (str): Identifier for hyperparameters.
        weights_path (str): Path to save model weights.
        diagrams_path (str): Path to save visualizations.
        launch_number (int): Launch number for this training run.
        visualize_every_xth_epoch (int): Visualize every Xth epoch.
        max_patience (int): Early stopping patience.
        model_type (str): Type of model (for naming files).
        gradient_clip (Optional[float]): Gradient clipping value. None to disable. Passed to loss_fn.
        control_fn (Optional[Callable]): Function for visualization/control.
                                        Signature: control_fn(models: Dict[str, nn.Module], 
                                                             train_batch: Tuple, 
                                                             val_batch: Tuple,
                                                             epoch: int,
                                                             diagrams_path: str,
                                                             hyperparams_id: str,
                                                             model_type: str,
                                                             launch_number: int) -> None
        early_stopping_metric (str): Name of metric to use for early stopping (default: "loss").

    Returns:
        Dict[str, Any]: Dictionary containing metrics object and training information.
    """
    current_patience = 0
    best_val_metric = None

    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(diagrams_path, exist_ok=True)

    for model_name, model in models.items():
        model_path = os.path.join(weights_path, f"{model_type}_{model_name}_hyperparams_{hyperparams_id}.pth")
        if os.path.exists(model_path):
            print(f"Loading existing weights for {model_name} from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No existing weights found for {model_name} at {model_path}. Starting fresh training.")
        model.to(device)

    first_batch_train = next(iter(train_loader))
    first_batch_val = next(iter(validation_loader))

    first_batch_train = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in first_batch_train)
    first_batch_val = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in first_batch_val)

    for epoch in range(num_epochs):
        metrics.setPhase('train')

        for model in models.values():
            model.train()

        train_iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} [Training]",
            leave=False
        )

        for batch in train_iterator:
            batch = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            loss_output = loss_fn(models, batch, metrics, gradient_clip)

            if 'loss' not in loss_output:
                raise ValueError("loss_fn must return a dict with 'loss' key for backpropagation")

            loss_tensor = loss_output['loss']

            for optimizer in optimizers.values():
                optimizer.step()

            loss_value = loss_tensor.item()
            train_iterator.set_postfix(loss=f"{loss_value:.6f}")

        if schedulers is not None:
            for scheduler in schedulers.values():
                scheduler.step()

        metrics.setPhase('val')

        for model in models.values():
            model.eval()

        val_iterator = tqdm(
            validation_loader,
            total=len(validation_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} [Validation]",
            leave=False
        )

        with torch.no_grad():
            for batch in val_iterator:
                batch = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)

                val_loss_output = loss_fn(models, batch, metrics, None)

                if 'loss' not in val_loss_output:
                    raise ValueError("loss_fn must return a dict with 'loss' key")

                val_loss_tensor = val_loss_output['loss']

                val_loss_value = val_loss_tensor.item()
                val_iterator.set_postfix(loss=f"{val_loss_value:.6f}")

        metrics.finalizeEpoch(epoch=epoch + 1)

        metrics.display(epoch=epoch + 1)

        if control_fn is not None and (epoch + 1) % visualize_every_xth_epoch == 0:
            with torch.no_grad():
                control_fn(
                    models=models,
                    train_batch=first_batch_train,
                    val_batch=first_batch_val,
                    epoch=epoch + 1,
                    diagrams_path=diagrams_path,
                    hyperparams_id=hyperparams_id,
                    model_type=model_type,
                    launch_number=launch_number
                )

        for model_name, model in models.items():
            model_path = os.path.join(weights_path, f"{model_type}_{model_name}_hyperparams_{hyperparams_id}.pth")
            torch.save(model.state_dict(), model_path)

        metrics.save(diagrams_data_path, hyperparams_id, model_type, launch_number)

        val_metric = metrics.getMetric(epoch + 1, 'val', early_stopping_metric)

        if best_val_metric is None or val_metric < best_val_metric:
            best_val_metric = val_metric
            current_patience = 0
        else:
            current_patience += 1

        if current_patience >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    return {
        "metrics": metrics,
        "best_val_metric": best_val_metric,
        "epochs_trained": epoch + 1
    }