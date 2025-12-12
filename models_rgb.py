import torch
import torch.nn as nn

from pyaiwrap.neural_network import ColorizationTransformerNet, ConvAttenColorizationNetwork, NeuralNetwork

def build_colorization_transformer_single_channel(
    image_size: int = 256,
    embed_dim: int = 512,
    num_heads: int = 8,
    mlp_ratio: int = 4,
    dropout: float = 0.1,
    num_layers: int = 6,
) -> ColorizationTransformerNet:
    """
    Tworzy pojedynczy model ColorizationTransformerNet do jednego kanału (R albo G albo B).
    """
    model = ColorizationTransformerNet(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        num_layers=num_layers,
        num_color_tokens=4096,      # możesz potem zgrać z DDColor
        num_image_patches=4096,     # jw.
        image_size=image_size,
        use_decoder_masking=False,
        only_use_encoder=False,
        output_channels=1           # KLUCZ: pojedynczy kanał
    )
    return model


def build_rgb_separate_models(
    image_size: int = 256,
) -> dict:
    """
    Zwraca słownik trzech osobnych modeli: R, G, B.
    """
    model_r = build_colorization_transformer_single_channel(image_size=image_size)
    model_g = build_colorization_transformer_single_channel(image_size=image_size)
    model_b = build_colorization_transformer_single_channel(image_size=image_size)

    return {
        "red_model": model_r,
        "green_model": model_g,
        "blue_model": model_b,
    }


def build_conv_atten_colorization_network(
    pretrained_channel_models: dict,
    trainable_network: nn.Module,
) -> ConvAttenColorizationNetwork:
    """
    Buduje ConvAttenColorizationNetwork z danymi modelami kanałów (R, G, B)
    i dodatkową siecią poprawiającą kolor.
    """
    # ConvAttenColorizationNetwork w repo przyjmuje konfigurację z plikami JSON i wagami,
    # ale my możemy go użyć trochę inaczej – bez plików – trzeba będzie go delikatnie przerobić,
    # lub stworzyć wariant, który przyjmuje już gotowe nn.Module.
    raise NotImplementedError("Za chwilę zrobimy wersję, która przyjmuje gotowe modele kanałów.")
