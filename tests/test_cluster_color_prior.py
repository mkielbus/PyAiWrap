"""The cluster colour prior: what travels in the checkpoint, and what it refuses to do."""
from __future__ import annotations

from typing import Dict

import pytest
import torch

from pyaiwrap.neural_network import (CLUSTER_PRIOR_COLORS, CLUSTER_PRIOR_STATISTICS,
                                     ClusterColorPrior, ColorMemoryTransformer)


FEATURE_DIM: int = 32
NUM_CLUSTERS: int = 6
PRIOR_DIM: int = len(CLUSTER_PRIOR_COLORS) * len(CLUSTER_PRIOR_STATISTICS)


def makeTables(seed: int = 0) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "feature_mean": torch.zeros(1, FEATURE_DIM),
        "feature_std": torch.ones(1, FEATURE_DIM),
        "head_weight": torch.randn(NUM_CLUSTERS, FEATURE_DIM, generator=generator),
        "head_bias": torch.zeros(NUM_CLUSTERS),
        "color_statistics": torch.randn(NUM_CLUSTERS, PRIOR_DIM, generator=generator),
    }


def makePrior(output_dim: int = 16) -> ClusterColorPrior:
    prior = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=output_dim,
                              num_clusters=NUM_CLUSTERS)
    prior.loadTables(**makeTables())
    return prior


def makeStages(batch: int = 2) -> list:
    """Four pyramid stages whose channels sum to FEATURE_DIM."""
    return [torch.rand(batch, 8, 4, 4) for _ in range(4)]


def test_tables_are_buffers_and_travel_in_state_dict() -> None:
    prior = makePrior()
    state = prior.state_dict()

    for name in ("feature_mean", "feature_std", "head_weight", "head_bias",
                 "color_statistics"):
        assert name in state

    restored = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                                 num_clusters=NUM_CLUSTERS)
    restored.load_state_dict(state)
    assert torch.equal(restored.head_weight, prior.head_weight)
    assert torch.equal(restored.color_statistics, prior.color_statistics)


def test_tables_are_not_trained() -> None:
    prior = makePrior()
    trainable = {name for name, parameter in prior.named_parameters()
                 if parameter.requires_grad}

    assert trainable == {"projection.0.weight", "projection.0.bias",
                         "projection.2.weight", "projection.2.bias"}


def test_forward_refuses_to_run_on_empty_tables() -> None:
    """Zeros would condition every image identically and look like a model ignoring the prior."""
    prior = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                              num_clusters=NUM_CLUSTERS)

    with pytest.raises(RuntimeError, match="no tables"):
        prior(makeStages())


def test_output_shape_and_batch_independence() -> None:
    prior = makePrior(output_dim=24).eval()
    stages = makeStages(batch=3)

    with torch.no_grad():
        output = prior(stages)
        single = prior([stage[1:2] for stage in stages])

    assert output.shape == (3, 24)
    assert torch.allclose(output[1:2], single, atol=1e-5)


def test_mixture_is_soft_not_argmax() -> None:
    """A confidently wrong cluster should degrade the hint, not replace it with a wrong row."""
    prior = makePrior().eval()
    prior.color_statistics.zero_()
    prior.color_statistics[0, 0] = 1.0
    prior.color_statistics[1, 0] = -1.0

    stages = makeStages(batch=4)
    with torch.no_grad():
        posterior = torch.softmax(
            torch.nn.functional.linear(
                torch.cat([stage.float().mean(dim=(2, 3)) for stage in stages], dim=1),
                prior.head_weight, prior.head_bias),
            dim=1)

    assert posterior.shape == (4, NUM_CLUSTERS)
    assert (posterior.max(dim=1).values < 1.0).all()
    assert torch.allclose(posterior.sum(dim=1), torch.ones(4), atol=1e-5)


def test_wrong_table_shape_rejected() -> None:
    prior = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                              num_clusters=NUM_CLUSTERS)
    tables = makeTables()
    tables["color_statistics"] = torch.zeros(NUM_CLUSTERS, PRIOR_DIM + 1)

    with pytest.raises(ValueError, match="color_statistics must have shape"):
        prior.loadTables(**tables)


def test_transformer_requires_a_pretrained_encoder() -> None:
    with pytest.raises(ValueError, match="needs a pretrained_encoder"):
        ColorMemoryTransformer(cluster_color_prior={"num_clusters": 4})


def test_transformer_without_prior_is_unchanged() -> None:
    """No prior configured must leave the colour queries exactly as before."""
    arguments = dict(color_dim=64, embed_dim=64, color_decoder_output_dim=64, num_heads=4,
                     mlp_ratio=2, dropout=0.0, color_decoder_layers=1, memory_size=8,
                     pretrained_encoder="convnext_tiny",
                     pretrained_encoder_stage_channels=[16, 32, 32, 64])

    model = ColorMemoryTransformer(**arguments)
    assert model.cluster_color_prior is None
    assert not any("cluster_color_prior" in key for key in model.state_dict())


def test_variant_mismatch_is_rejected() -> None:
    """Tiny and Small share stage widths, so shapes alone cannot catch a swapped backbone."""
    prior = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                              num_clusters=NUM_CLUSTERS, backbone_variant="small")

    with pytest.raises(ValueError, match="built on ConvNeXt-tiny"):
        prior.loadTables(**makeTables(), variant="tiny")


def test_matching_variant_accepted() -> None:
    prior = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                              num_clusters=NUM_CLUSTERS, backbone_variant="small")
    prior.loadTables(**makeTables(), variant="small")

    assert bool(prior.head_weight.any())


def test_unlabelled_tables_still_load() -> None:
    """Tables written before the variant was recorded must not become unusable."""
    prior = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                              num_clusters=NUM_CLUSTERS, backbone_variant="small")
    prior.loadTables(**makeTables())

    assert bool(prior.head_weight.any())


def buildUnetLayers(bottleneck_channels: int = 16) -> list:
    """Two downsampling stages, so the bottleneck sits at 1/4 and each decoder block's skip
    lands at the resolution that block produces."""
    def encoder(name: str, in_channels: int, out_channels: int) -> Dict[str, object]:
        return {"type": "UNetEncoderBlock", "params": {
            "in_channels": in_channels, "out_channels": out_channels,
            "downsample": True, "block_name": name, "norm_groups": 8}}

    def decoder(name: str, skip: str) -> Dict[str, object]:
        return {"type": "UNetDecoderBlock", "params": {
            "in_channels": bottleneck_channels, "out_channels": bottleneck_channels,
            "upsample": True, "skip_connection": skip, "block_name": name, "norm_groups": 8}}

    return [
        encoder("enc1", 1, bottleneck_channels),
        encoder("enc2", bottleneck_channels, bottleneck_channels),
        {"type": "UNetBottleneck", "params": {"channels": bottleneck_channels,
                                              "bottleneck_channels": bottleneck_channels,
                                              "norm_groups": 8}},
        decoder("dec1", "enc2"),
        decoder("dec2", "enc1"),
    ]


def test_unet_prior_requires_the_multi_scale_encoder() -> None:
    from pyaiwrap.neural_network import UNetWithSkipConnections

    with pytest.raises(ValueError, match="needs multi_scale_semantic_encoder"):
        UNetWithSkipConnections(buildUnetLayers(),
                                cluster_color_prior={"num_clusters": NUM_CLUSTERS})


def test_unet_film_starts_as_the_identity() -> None:
    """A fresh conditioned model must start exactly where the unconditioned one does."""
    from pyaiwrap.neural_network import FilmModulation

    film = FilmModulation(condition_dim=8, channels=4)
    features = torch.rand(2, 4, 3, 3)

    assert torch.allclose(film(features, torch.randn(2, 8)), features)


def test_unet_prior_conditions_the_bottleneck() -> None:
    from pyaiwrap.neural_network import UNetWithSkipConnections

    network = UNetWithSkipConnections(
        buildUnetLayers(),
        multi_scale_semantic_encoder={"out_channels": 16, "output_strides": [4],
                                      "norm_groups": 8, "pretrained": False},
        cluster_color_prior={"num_clusters": NUM_CLUSTERS},
        semantic_input_channel=0
    )
    network.cluster_color_prior.loadTables(
        feature_mean=torch.zeros(1, 1440), feature_std=torch.ones(1, 1440),
        head_weight=torch.randn(NUM_CLUSTERS, 1440), head_bias=torch.zeros(NUM_CLUSTERS),
        color_statistics=torch.randn(NUM_CLUSTERS, PRIOR_DIM),
    )

    assert network.prior_film is not None
    with torch.no_grad():
        assert network(torch.rand(2, 1, 32, 32)).shape[0] == 2


def test_backbone_runs_once_for_pyramid_and_prior() -> None:
    """Running the frozen backbone twice would double the branch's only expensive part."""
    from pyaiwrap.neural_network import UNetWithSkipConnections

    network = UNetWithSkipConnections(
        buildUnetLayers(),
        multi_scale_semantic_encoder={"out_channels": 16, "output_strides": [4],
                                      "norm_groups": 8, "pretrained": False},
        cluster_color_prior={"num_clusters": NUM_CLUSTERS},
        semantic_input_channel=0
    )
    network.cluster_color_prior.loadTables(
        feature_mean=torch.zeros(1, 1440), feature_std=torch.ones(1, 1440),
        head_weight=torch.randn(NUM_CLUSTERS, 1440), head_bias=torch.zeros(NUM_CLUSTERS),
        color_statistics=torch.randn(NUM_CLUSTERS, PRIOR_DIM),
    )

    backbone = network.multi_scale_semantic_encoder.backbone
    calls = {"count": 0}
    original = backbone.forward

    def counting(luminance):
        calls["count"] += 1
        return original(luminance)

    backbone.forward = counting
    with torch.no_grad():
        network(torch.rand(1, 1, 32, 32))

    assert calls["count"] == 1


def test_missing_tables_file_is_tolerated_at_build_time(tmp_path) -> None:
    """Inference builds the architecture and then overwrites it from the checkpoint, so a
    build-time file must not be a deployment dependency."""
    with pytest.warns(RuntimeWarning, match="not found"):
        prior = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                                  num_clusters=NUM_CLUSTERS,
                                  tables_path=str(tmp_path / "absent.pt"))

    assert not bool(prior.head_weight.any())
    with pytest.raises(RuntimeError, match="no tables"):
        prior(makeStages())

    seeded = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                               num_clusters=NUM_CLUSTERS)
    seeded.loadTables(**makeTables())
    prior.load_state_dict(seeded.state_dict())

    assert bool(prior.head_weight.any())
    with torch.no_grad():
        assert prior(makeStages()).shape == (2, 16)


def test_present_tables_file_is_used(tmp_path) -> None:
    path = tmp_path / "tables.pt"
    tables = makeTables()
    torch.save({**tables, "variant": "small"}, path)

    prior = ClusterColorPrior(feature_dim=FEATURE_DIM, output_dim=16,
                              num_clusters=NUM_CLUSTERS, backbone_variant="small",
                              tables_path=str(path))

    assert torch.equal(prior.head_weight, tables["head_weight"])
