import torch
from typing import Dict, Any


def createScheduler(optimizer, hyperparams: Dict[str, Any], train_loader_len: int):
    """
    Create learning rate scheduler based on hyperparameters.

    Args:
        optimizer: The optimizer to schedule
        hyperparams: Dictionary of hyperparameters
        train_loader_len: Length of train loader (steps per epoch)
        epochs: Total number of epochs

    Returns:
        Configured learning rate scheduler
    """
    scheduler_type = hyperparams.get("SCHEDULER_TYPE", "exponential")
    learning_rate = hyperparams.get("LEARNING_RATE", 0.0001)
    min_lr = hyperparams.get("MIN_LR", 1e-6)
    gamma = hyperparams.get("GAMMA", 0.99)

    if scheduler_type == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hyperparams.get("T_0", 30),
            T_mult=hyperparams.get("T_MULT", 2),
            eta_min=min_lr
        )

    elif scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * hyperparams.get("MAX_LR_MULTIPLIER", 10),
            epochs=hyperparams.get("EPOCHS", 100),
            steps_per_epoch=train_loader_len,
            pct_start=hyperparams.get("PCT_START", 0.1),
            div_factor=hyperparams.get("DIV_FACTOR", 10),
            final_div_factor=hyperparams.get("FINAL_DIV_FACTOR", 100)
        )

    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hyperparams.get("EPOCHS", 100),
            eta_min=min_lr
        )

    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=hyperparams.get("STEP_SIZE", 30),
            gamma=hyperparams.get("STEP_GAMMA", 0.1)
        )

    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
    return scheduler
