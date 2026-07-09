import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from itertools import islice
from typing import Any, Dict, Callable, Optional, Tuple
from .metrics import Metrics


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
    config_id: str = "0",
    weights_path: str = "./weights",
    diagrams_path: str = "./diagrams",
    launch_number: int = 0,
    visualize_every_xth_epoch: int | None = 5,
    max_patience: int = 15,
    model_type: str = "generator",
    gradient_clip: Optional[float] = 1.0,
    control_fn: Optional[Callable] = None,
    early_stopping_metric: str = "loss",
    control_train_batch_number: int = 0,
    control_val_batch_number: int = 0
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
        control_train_batch_number (int): Index of the train batch passed to control_fn (default: 0).
        control_val_batch_number (int): Index of the validation batch passed to control_fn (default: 0).

    Returns:
        Dict[str, Any]: Dictionary containing metrics object and training information.
    """
    current_patience = 0
    best_val_metric = None
    early_stopping_triggered = False
    start_epoch = 0

    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(diagrams_path, exist_ok=True)

    checkpoint_path = os.path.join(weights_path, f"{model_type}_training_state_hyperparams_{config_id}.pth")

    for model_name, model in models.items():
        model_path = os.path.join(weights_path, f"{model_type}_{model_name}_hyperparams_{config_id}.pth")

        if os.path.exists(model_path):
            print(f"Loading existing weights for {model_name} from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No existing weights found for {model_name} at {model_path}. Starting fresh training.")

        model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading training state from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        for model_name, optimizer in optimizers.items():
            if model_name in checkpoint['optimizers']:
                optimizer.load_state_dict(checkpoint['optimizers'][model_name])
                print(f"Loaded optimizer state for {model_name}")

        if schedulers is not None:
            for model_name, scheduler in schedulers.items():
                if model_name in checkpoint.get('schedulers', {}):
                    scheduler.load_state_dict(checkpoint['schedulers'][model_name])
                    print(f"Loaded scheduler state for {model_name}")

        start_epoch = checkpoint.get('epoch', 0)
        best_val_metric = checkpoint.get('best_val_metric', None)
        current_patience = checkpoint.get('current_patience', 0)

        if 'metrics_state' in checkpoint and hasattr(metrics, 'setState'):
            metrics.setState(checkpoint['metrics_state'])
            print("Loaded metrics history")

        print(f"Resuming training from epoch {start_epoch}")

    if control_train_batch_number >= len(train_loader):
        raise ValueError(f"control_train_batch_number ({control_train_batch_number}) is out of range "
                         f"for train_loader with {len(train_loader)} batches")
    if control_val_batch_number >= len(validation_loader):
        raise ValueError(f"control_val_batch_number ({control_val_batch_number}) is out of range "
                         f"for validation_loader with {len(validation_loader)} batches")

    control_batch_train = next(islice(iter(train_loader), control_train_batch_number, None))
    control_batch_val = next(islice(iter(validation_loader), control_val_batch_number, None))

    control_batch_train = tuple(
        item.detach().clone().to(device) if isinstance(item, torch.Tensor) else copy.deepcopy(item)
        for item in control_batch_train
    )
    control_batch_val = tuple(
        item.detach().clone().to(device) if isinstance(item, torch.Tensor) else copy.deepcopy(item)
        for item in control_batch_val
    )

    # If resuming a run that already reached num_epochs, the loop below never
    # executes; `epoch` should be defined so the return statement works
    epoch = start_epoch - 1

    epoch_iterator = tqdm(
            range(start_epoch, num_epochs),
            total=num_epochs - start_epoch,
            desc="Epochs passed",
            leave=False,
            position=0
        )

    for epoch in epoch_iterator:
        metrics.setPhase('train')

        for model in models.values():
            model.train()

        train_iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc="[Training]",
            leave=False,
            position=1
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
            desc="[Validation]",
            leave=False,
            position=1
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

        if control_fn is not None and visualize_every_xth_epoch and (epoch + 1) % visualize_every_xth_epoch == 0:
            with torch.no_grad():
                control_fn(
                    models=models,
                    train_batch=control_batch_train,
                    val_batch=control_batch_val,
                    epoch=epoch + 1,
                    diagrams_path=diagrams_path,
                    config_id=config_id,
                    model_type=model_type,
                    launch_number=launch_number
                )

        metrics.save(diagrams_data_path, config_id, model_type, launch_number)

        for model_name, model in models.items():
            model_path = os.path.join(weights_path, f"{model_type}_{model_name}_hyperparams_{config_id}.pth")
            torch.save(model.state_dict(), model_path)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_metric': best_val_metric,
            'current_patience': current_patience,
            'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
        }

        if schedulers is not None:
            checkpoint['schedulers'] = {name: scheduler.state_dict() for name, scheduler in schedulers.items()}

        if hasattr(metrics, 'getState'):
            checkpoint['metrics_state'] = metrics.getState()

        torch.save(checkpoint, checkpoint_path)

        val_metric = metrics.getMetric(epoch + 1, 'val', early_stopping_metric)

        if best_val_metric is None or val_metric < best_val_metric:
            best_val_metric = val_metric
            current_patience = 0
            for model_name, model in models.items():
                best_model_path = os.path.join(weights_path, f"best_performance_{model_type}_{model_name}_hyperparams_{config_id}.pth")
                torch.save(model.state_dict(), best_model_path)

            best_checkpoint = {
                'epoch': epoch + 1,
                'best_val_metric': best_val_metric,
                'current_patience': current_patience,
                'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
            }

            if schedulers is not None:
                best_checkpoint['schedulers'] = {name: scheduler.state_dict() for name, scheduler in schedulers.items()}

            if hasattr(metrics, 'getState'):
                best_checkpoint['metrics_state'] = metrics.getState()

            best_checkpoint_path = os.path.join(weights_path, f"best_performance_{model_type}_training_state_hyperparams_{config_id}.pth")
            torch.save(best_checkpoint, best_checkpoint_path)
        else:
            current_patience += 1

        if current_patience >= max_patience:
            early_stopping_triggered = True
            break

    return {
        "metrics": metrics,
        "best_val_metric": best_val_metric,
        "epochs_trained": epoch + 1,
        'early_stopped': early_stopping_triggered
    }
