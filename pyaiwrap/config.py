from typing import Any, Dict, List
import json
from .neural_network import NeuralNetwork


def loadLayersFromJson(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSON file containing an array of layer configs."""
    with open(file_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, list):
        raise TypeError("JSON must contain a list of layer definitions.")
    return data


def buildNeuralNetworkFromJson(file_path: str) -> NeuralNetwork:
    layers = loadLayersFromJson(file_path)
    return NeuralNetwork(layers)
