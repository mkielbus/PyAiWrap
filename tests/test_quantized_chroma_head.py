"""The binned chroma head: continuous decoding, soft targets and class rebalancing."""
from __future__ import annotations

import math

import pytest
import torch

from pyaiwrap.neural_network import QuantizedChromaHead


NUM_BINS: int = 9
BIN_SIZE: float = 10.0


def makePalette() -> dict:
    """A 3x3 grid of cell centres, so the geometry is checkable by hand."""
    coordinates = torch.tensor([-BIN_SIZE, 0.0, BIN_SIZE])
    centres = torch.stack([
        coordinates.repeat_interleave(3),
        coordinates.repeat(3)
    ], dim=1)
    frequencies = torch.full((NUM_BINS,), 1.0 / NUM_BINS)
    frequencies[4] = 0.5  # the neutral cell dominates, as it does in this corpus
    frequencies = frequencies / frequencies.sum()
    return {"bin_centres": centres, "bin_frequencies": frequencies, "bin_size": BIN_SIZE}


def makeHead(**kwargs) -> QuantizedChromaHead:
    head = QuantizedChromaHead(in_channels=4, num_bins=NUM_BINS, **kwargs)
    head.loadBins(**makePalette())
    return head


def test_palette_travels_in_state_dict() -> None:
    head = makeHead()
    state = head.state_dict()

    for name in ("bin_centres", "bin_frequencies", "class_weights"):
        assert name in state

    restored = QuantizedChromaHead(in_channels=4, num_bins=NUM_BINS)
    restored.load_state_dict(state)
    assert torch.equal(restored.bin_centres, head.bin_centres)


def test_forward_refuses_without_a_palette() -> None:
    head = QuantizedChromaHead(in_channels=4, num_bins=NUM_BINS)

    with pytest.raises(RuntimeError, match="no palette"):
        head(torch.rand(1, 4, 2, 2))


def test_decoding_is_continuous_not_snapped() -> None:
    """The whole objection to binning is that the output is confined to the palette. It is not:
    the annealed mean is a weighted average of centres and lands between them."""
    head = makeHead(temperature=1.0)
    logits = torch.full((1, NUM_BINS, 1, 1), -20.0)
    logits[0, 0] = 0.0   # centre (-10, -10)
    logits[0, 1] = 0.0   # centre (-10, 0)

    decoded = head.decode(logits)

    assert torch.allclose(decoded[0, :, 0, 0], torch.tensor([-10.0, -5.0]), atol=1e-3)
    assert not any(torch.allclose(decoded[0, :, 0, 0], centre, atol=1e-3)
                   for centre in head.bin_centres)


def test_low_temperature_approaches_the_mode() -> None:
    head = makeHead(temperature=0.05)
    logits = torch.full((1, NUM_BINS, 1, 1), 0.0)
    logits[0, 8] = 1.0  # centre (10, 10)

    decoded = head.decode(logits)

    assert torch.allclose(decoded[0, :, 0, 0], torch.tensor([10.0, 10.0]), atol=0.5)


def test_soft_encoding_is_a_distribution_centred_on_the_truth() -> None:
    """Returned sparsely -- indices and weights -- because the dense form is 0.9 GiB at the
    resolution and batch this runs at."""
    head = makeHead()
    chroma = torch.zeros(1, 2, 1, 1)

    indices, weights = head.softEncode(chroma)

    assert indices.shape == (1, head.encode_neighbours)
    assert math.isclose(float(weights.sum()), 1.0, abs_tol=1e-5)
    assert int(indices[0, int(weights[0].argmax())]) == 4  # the (0, 0) cell


def test_soft_encoding_beats_one_hot_on_near_misses() -> None:
    """A near miss must cost less than the opposite side of the wheel, which one-hot cannot do."""
    head = makeHead()
    chroma = torch.tensor([[[[2.0]], [[2.0]]]])

    indices, weights = head.softEncode(chroma)
    by_bin = {int(index): float(weight) for index, weight in zip(indices[0], weights[0])}

    assert by_bin[4] > by_bin[8] > 0.0
    assert by_bin.get(0, 0.0) < by_bin[8]


def test_chunked_encoding_matches_unchunked() -> None:
    """Chunking is a memory measure and must not change a single weight."""
    head = makeHead()
    chroma = torch.randn(2, 2, 8, 8) * 8.0

    head.encode_chunk = 10 ** 9
    whole_indices, whole_weights = head.softEncode(chroma)
    head.encode_chunk = 7
    chunked_indices, chunked_weights = head.softEncode(chroma)

    assert torch.equal(whole_indices, chunked_indices)
    assert torch.allclose(whole_weights, chunked_weights)


def test_target_follows_the_head_down_an_octave() -> None:
    """With upsample_factor the logits are smaller than the image; the target is resized to the
    logits rather than the logits being upsampled, which would waste the saving."""
    head = makeHead(upsample_factor=2)
    logits = torch.randn(1, NUM_BINS, 4, 4)
    chroma = torch.randn(1, 2, 8, 8) * 5.0

    assert float(head.classificationLoss(logits, chroma)) > 0.0


def test_upsampling_head_returns_image_resolution() -> None:
    head = makeHead(upsample_factor=2)

    assert head(torch.rand(1, 4, 4, 4)).shape == (1, 2, 8, 8)


def test_class_weights_favour_rare_colours_without_changing_scale() -> None:
    head = makeHead()
    frequencies = head.bin_frequencies

    assert head.class_weights[4] < head.class_weights[0]
    assert math.isclose(float((frequencies * head.class_weights).sum()), 1.0, abs_tol=1e-4)


def test_rebalancing_can_be_switched_off() -> None:
    head = makeHead(rebalance_lambda=1.0)

    assert torch.allclose(head.class_weights, head.class_weights[0].expand(NUM_BINS),
                          atol=1e-5)


def test_classification_loss_is_lowest_for_the_truth() -> None:
    head = makeHead()
    chroma = torch.zeros(2, 2, 3, 3)

    confident_correct = torch.full((2, NUM_BINS, 3, 3), -10.0)
    confident_correct[:, 4] = 10.0
    confident_wrong = torch.full((2, NUM_BINS, 3, 3), -10.0)
    confident_wrong[:, 8] = 10.0

    assert (head.classificationLoss(confident_correct, chroma)
            < head.classificationLoss(confident_wrong, chroma))


def test_invalid_temperature_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        QuantizedChromaHead(in_channels=4, num_bins=NUM_BINS, temperature=0.0)


def test_palette_size_mismatch_names_the_cause() -> None:
    head = QuantizedChromaHead(in_channels=4, num_bins=NUM_BINS + 1)

    with pytest.raises(ValueError, match="num_bins in the architecture"):
        head.loadBins(**makePalette())


def test_missing_palette_file_is_tolerated_at_build_time(tmp_path) -> None:
    with pytest.warns(RuntimeWarning, match="not found"):
        head = QuantizedChromaHead(in_channels=4, num_bins=NUM_BINS,
                                   bins_path=str(tmp_path / "absent.pt"))

    assert not bool(head.bin_centres.any())


def test_bare_transformer_constructs_and_runs() -> None:
    """The default arguments used to disagree with each other: color_dim 256 against embed_dim
    512, which passed construction and then failed at the first forward pass."""
    from pyaiwrap.neural_network import ColorMemoryTransformer

    model = ColorMemoryTransformer()
    with torch.no_grad():
        assert model(torch.rand(1, 1, 64, 64)).shape[0] == 1


@pytest.mark.parametrize("color_dim,embed_dim", [(256, 512), (128, 256)])
def test_decoder_rejects_mismatched_query_and_attention_widths(color_dim: int,
                                                               embed_dim: int) -> None:
    """Reported where it can be fixed, not as a shape error deep in the first forward."""
    from pyaiwrap.neural_network import MultiScaleColorDecoder

    with pytest.raises(ValueError, match="must be equal"):
        MultiScaleColorDecoder(color_dim=color_dim, embed_dim=embed_dim, output_dim=embed_dim)


def test_decoder_output_width_stays_free() -> None:
    """output_dim only sizes the final projection; the constraint tying it to embed_dim is the
    consumer's, and existing tests rely on being able to vary it."""
    from pyaiwrap.neural_network import MultiScaleColorDecoder

    decoder = MultiScaleColorDecoder(color_dim=32, embed_dim=32, output_dim=16,
                                     num_heads=4, num_layers=1, memory_size=8)
    assert decoder._final_projection.out_features == 16


def test_transformer_rejects_an_output_width_it_cannot_contract() -> None:
    from pyaiwrap.neural_network import ColorMemoryTransformer

    with pytest.raises(ValueError, match="color_decoder_output_dim"):
        ColorMemoryTransformer(color_dim=128, embed_dim=128, color_decoder_output_dim=64)
