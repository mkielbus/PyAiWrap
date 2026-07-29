"""Tests for the AdamW weight-decay split (createParameterGroups) and query init.

The invariant that matters: decay reaches the matmul/conv weights and nothing else. Norm
gains, biases and embedding tables must land in a group with weight_decay == 0, because
decaying a weakly-gradient embedding pins it at its initial value -- which is how the v5
color memory stayed at ~zero for 110 epochs.

Naming/style follows the project convention (see CLAUDE.md).
"""
from typing import Dict, List

import torch
import torch.nn as nn

from pyaiwrap.neural_network import QUERY_EMBEDDING_INIT_STD, MultiScaleColorDecoder
from pyaiwrap.optimizers import createOptimizer, createParameterGroups


class _MixedNet(nn.Module):
    """One of each parameter kind the split has to classify."""

    def __init__(self) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(2, 4, 3, bias=True)
        self.linear: nn.Linear = nn.Linear(4, 4, bias=True)
        self.norm: nn.LayerNorm = nn.LayerNorm(4)
        self.embedding: nn.Embedding = nn.Embedding(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(self.conv(x).flatten(1)))


def _groupOf(groups: List[Dict], parameter: nn.Parameter) -> Dict:
    for group in groups:
        if any(candidate is parameter for candidate in group["params"]):
            return group
    raise AssertionError("parameter missing from every group")


def testMatmulWeightsKeepDecay() -> None:
    model = _MixedNet()
    groups = createParameterGroups(model, weight_decay=0.05)

    assert _groupOf(groups, model.conv.weight)["weight_decay"] == 0.05
    assert _groupOf(groups, model.linear.weight)["weight_decay"] == 0.05


def testNormsBiasesAndEmbeddingsAreExcluded() -> None:
    model = _MixedNet()
    groups = createParameterGroups(model, weight_decay=0.05)

    for parameter in (model.conv.bias, model.linear.bias,
                      model.norm.weight, model.norm.bias,
                      model.embedding.weight):
        assert _groupOf(groups, parameter)["weight_decay"] == 0.0


def testEveryTrainableParameterAppearsExactlyOnce() -> None:
    model = _MixedNet()
    groups = createParameterGroups(model, weight_decay=0.05)

    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == len(list(model.parameters()))
    assert all(sum(candidate is parameter for candidate in grouped) == 1
               for parameter in model.parameters())


def testFrozenParametersAreNotOptimized() -> None:
    model = _MixedNet()
    model.conv.weight.requires_grad_(False)
    groups = createParameterGroups(model, weight_decay=0.05)

    grouped = [parameter for group in groups for parameter in group["params"]]
    assert all(candidate is not model.conv.weight for candidate in grouped)


def testEmptyGroupsAreDropped() -> None:
    model = nn.LayerNorm(4)  # only ndim <= 1 parameters, so the decay group is empty
    groups = createParameterGroups(model, weight_decay=0.05)

    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.0


def testAdamWReceivesTheSplitWhenGivenAModule() -> None:
    model = _MixedNet()
    config = {"OPTIMIZER_TYPE": "adamw", "LEARNING_RATE": 1e-4,
              "WEIGHT_DECAY": 0.05, "B1": 0.9, "B2": 0.999}

    optimizer = createOptimizer(model, config)

    assert len(optimizer.param_groups) == 2
    assert sorted(group["weight_decay"] for group in optimizer.param_groups) == [0.0, 0.05]


def testParameterIteratorKeepsSingleGroupBehaviour() -> None:
    model = _MixedNet()
    config = {"OPTIMIZER_TYPE": "adamw", "LEARNING_RATE": 1e-4,
              "WEIGHT_DECAY": 0.05, "B1": 0.9, "B2": 0.999}

    optimizer = createOptimizer(model.parameters(), config)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.05


def testColorQueriesStartDistinct() -> None:
    """Zero init made every query identical; a forward pass could not tell them apart."""
    decoder = MultiScaleColorDecoder(color_dim=32, embed_dim=32, output_dim=16,
                                     num_heads=4, num_layers=2, memory_size=64)

    for table in (decoder.color_embeddings.weight, decoder.memory_embeddings.weight):
        assert table.std().item() > 0.0
        assert torch.cdist(table, table).max().item() > 0.0
        # Within a factor of two of the requested scale, i.e. not zero and not exploding.
        assert 0.5 * QUERY_EMBEDDING_INIT_STD < table.std().item() < 2.0 * QUERY_EMBEDDING_INIT_STD


def testColorQueriesSpanManyDirections() -> None:
    """Effective rank of the query table: v5 collapsed to ~13 of 128 after training."""
    decoder = MultiScaleColorDecoder(color_dim=64, embed_dim=64, output_dim=16,
                                     num_heads=4, num_layers=2, memory_size=64)

    singular_values = torch.linalg.svdvals(decoder.color_embeddings.weight.detach())
    effective_rank = (singular_values.sum() ** 2 / (singular_values ** 2).sum()).item()
    assert effective_rank > 32.0
