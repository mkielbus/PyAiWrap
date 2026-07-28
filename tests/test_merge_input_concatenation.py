"""Tests for feeding the pristine input into the trainable merge network.

Two invariants matter here. First, the historical per-layout behaviour must survive: a
chroma-only (a, b) model still gets luminance stacked on, an RGB model built without the
new flag still does not, so existing checkpoints keep loading. Second, when the flag is
set explicitly it must win for either layout, and the trainable network must receive the
input channels first (the order the merge architectures are wired for).

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

from pyaiwrap.neural_network import ConvAttenColorizationNetwork


class _ChannelRecorder(nn.Module):
    """Stand-in for the merge UNet: records what it was handed, returns a fixed shape."""

    def __init__(self, out_channels: int = 3) -> None:
        super().__init__()
        self.received: torch.Tensor = torch.empty(0)
        self._out_channels: int = out_channels
        # A parameter so the module has something for the optimizer to see.
        self.scale: nn.Parameter = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.received = x
        batch, _, height, width = x.shape
        return torch.zeros(batch, self._out_channels, height, width) * self.scale


class _ConstantChannel(nn.Module):
    """Frozen submodule stand-in emitting one channel filled with `value`."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value: float = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[:, :1], self._value)


def _buildNetwork(model_values: Dict[str, float], concatenate_input=None) -> ConvAttenColorizationNetwork:
    """Build the modular network with stubbed submodules (no weight files touched)."""
    network: ConvAttenColorizationNetwork = ConvAttenColorizationNetwork.__new__(ConvAttenColorizationNetwork)
    nn.Module.__init__(network)

    names: List[str] = network._resolve_model_names(model_values.keys())
    network._color_model_names = names
    network._pretrained_models_config = {}
    network._pretrained_input_channels = 1
    network._concatenate_input = network._resolveConcatenateInput(concatenate_input)
    network._pretrained_models = nn.ModuleDict(
        {name: _ConstantChannel(model_values[name]) for name in names}
    )
    network._trainable_network = _ChannelRecorder()
    return network


RGB_MODELS: Dict[str, float] = {"red_model": 0.1, "green_model": 0.2, "blue_model": 0.3}
LAB_MODELS: Dict[str, float] = {"a_model": 0.4, "b_model": 0.5}


def testRgbLayoutDefaultsToNoInputConcatenation() -> None:
    """The pre-existing RGB behaviour: the merge net sees only the three predictions."""
    network: ConvAttenColorizationNetwork = _buildNetwork(RGB_MODELS)
    network(torch.rand(2, 1, 8, 8))
    assert network._trainable_network.received.shape[1] == 3


def testLabLayoutDefaultsToInputConcatenation() -> None:
    """Chroma-only predictions are meaningless without luminance, so it is stacked on."""
    network: ConvAttenColorizationNetwork = _buildNetwork(LAB_MODELS)
    network(torch.rand(2, 1, 8, 8))
    assert network._trainable_network.received.shape[1] == 3


def testRgbLayoutConcatenatesInputWhenEnabled() -> None:
    network: ConvAttenColorizationNetwork = _buildNetwork(RGB_MODELS, concatenate_input=True)
    luminance: torch.Tensor = torch.rand(2, 1, 8, 8)
    network(luminance)

    received: torch.Tensor = network._trainable_network.received
    assert received.shape[1] == 4
    # Input first, then the submodules in their declared order.
    assert torch.equal(received[:, 0:1], luminance)
    assert torch.allclose(received[:, 1], torch.full((2, 8, 8), 0.1))
    assert torch.allclose(received[:, 2], torch.full((2, 8, 8), 0.2))
    assert torch.allclose(received[:, 3], torch.full((2, 8, 8), 0.3))


def testLabLayoutSkipsInputWhenDisabled() -> None:
    """An explicit False overrides the layout default in the other direction too."""
    network: ConvAttenColorizationNetwork = _buildNetwork(LAB_MODELS, concatenate_input=False)
    network(torch.rand(2, 1, 8, 8))
    assert network._trainable_network.received.shape[1] == 2


@pytest.mark.parametrize("models", [RGB_MODELS, LAB_MODELS])
def testInputStaysPristine(models: Dict[str, float]) -> None:
    """Whatever is concatenated is the untouched input, never a submodule's view of it."""
    network: ConvAttenColorizationNetwork = _buildNetwork(models, concatenate_input=True)
    luminance: torch.Tensor = torch.rand(2, 1, 8, 8)
    before: torch.Tensor = luminance.clone()
    network(luminance)

    assert torch.equal(luminance, before)
    assert torch.equal(network._trainable_network.received[:, 0:1], before)
