from typing import Dict, Any
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
        "EPOCHS": 100,
        "DIAGRAMS_DATA_PATH": "./diagrams_data",
        "WEIGHTS_PATH": "./weights",
        "PATIENCE": 15,
        "DIAGRAMS_PATH": "./diagrams",
        "VISUALIZE_EVERY": 10,
        "GRADIENT_CLIP": 1.0
    }

    for key, default_value in defaults.items():
        hyperparams.setdefault(key, default_value)

    return hyperparams
