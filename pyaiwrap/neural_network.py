from typing import Any, Dict, List, Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (convnext_tiny, ConvNeXt_Tiny_Weights,
                                convnext_small, ConvNeXt_Small_Weights,
                                convnext_base, ConvNeXt_Base_Weights)
import sys
import math
import json
import os
import warnings
from .utils import sinusoidalPositionalEncoding2D


class UnsupportedLayerType(Exception):
    """Raised when a layer type specified in the configuration is not supported."""
    def __init__(self, layer_type: str):
        super().__init__(f"Unsupported layer type: {layer_type}")
        self._layer_type = layer_type


class NotSupportedLayerConstructorParam(Exception):
    """Raised when a parameter in the configuration is not supported by the underlying layer."""
    def __init__(self, layer_type: str, param_name: str):
        super().__init__(f"Parameter '{param_name}' is not supported by layer type '{layer_type}'")
        self._layer_type = layer_type
        self._param_name = param_name


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection"""
    def __init__(self, channels: int, kernel_size: int = 3, use_batch_norm: bool = True):
        super().__init__()
        padding = kernel_size // 2

        convolution_layers = []
        convolution_layers.append(nn.Conv2d(channels, channels, kernel_size, padding=padding))
        if use_batch_norm:
            convolution_layers.append(nn.BatchNorm2d(channels))
        convolution_layers.append(nn.LeakyReLU(0.2, inplace=True))
        convolution_layers.append(nn.Conv2d(channels, channels, kernel_size, padding=padding))
        if use_batch_norm:
            convolution_layers.append(nn.BatchNorm2d(channels))

        self.convolution_block = nn.Sequential(*convolution_layers)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        residual_output = self.convolution_block(input_tensor)
        return self.activation(input_tensor + residual_output)


class DownsamplingBlock(nn.Module):
    """Downsampling block with residual connection"""
    def __init__(self, input_channels: int, output_channels: int, num_residual_blocks: int = 2):
        super().__init__()
        self.downsampling_convolution = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.batch_normalization = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(output_channels) for _ in range(num_residual_blocks)
        ])

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        downsampled_tensor = self.downsampling_convolution(input_tensor)
        normalized_tensor = self.batch_normalization(downsampled_tensor)
        activated_tensor = self.activation(normalized_tensor)
        output_tensor = self.residual_blocks(activated_tensor)
        return output_tensor


class UpsamplingBlock(nn.Module):
    """Upsampling block with residual connection"""
    def __init__(self, input_channels: int, output_channels: int, num_residual_blocks: int = 2):
        super().__init__()
        self.upsampling_convolution = nn.ConvTranspose2d(
            input_channels, output_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.batch_normalization = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=True)

        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(output_channels, use_batch_norm=True) for _ in range(num_residual_blocks)
        ])

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        upsampled_tensor = self.upsampling_convolution(input_tensor)
        normalized_tensor = self.batch_normalization(upsampled_tensor)
        activated_tensor = self.activation(normalized_tensor)
        output_tensor = self.residual_blocks(activated_tensor)
        return output_tensor


class ResizeCustom(nn.Module):
    """
    Custom resize layer that resizes input to (new_size x new_size)

    Args:
        new_size (int): Target size for both height and width
        mode (str): Interpolation mode ('bilinear', 'nearest', 'bicubic')
        align_corners (bool): Whether to align corners (for bilinear/bicubic)
    """
    def __init__(self, new_size=156, mode='bicubic', align_corners=False):
        super().__init__()
        self._new_size = new_size
        self._mode = mode
        self._align_corners = align_corners if mode in ['bilinear', 'bicubic'] else None

    def forward(self, x):
        return F.interpolate(
            x,
            size=(self._new_size, self._new_size),
            mode=self._mode,
            align_corners=self._align_corners
        )

    def extra_repr(self):
        return f'new_size={self._new_size}, mode={self._mode}, align_corners={self._align_corners}'


class MultiHeadAttention(nn.Module):
    """
    Generic Multi-Head Attention supporting:
    - Self-attention (Q, K, V from same source)
    - Cross-attention with separate K and V sources
    Only supports causal masking.
    """
    def __init__(self, query_dim: int, key_dim: Optional[int] = None,
                 value_dim: Optional[int] = None, embed_dim: Optional[int] = None,
                 num_heads: int = 8, dropout: float = 0.1,
                 use_causal_mask: bool = False):
        """
        Args:
            query_dim: Dimension of query input
            key_dim: Dimension of key input. If None, uses query_dim
            value_dim: Dimension of value input. If None, uses key_dim
            embed_dim: Internal embedding dimension. If None, uses query_dim
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_causal_mask: If True, apply causal masking
        """
        super().__init__()

        # Set defaults
        self.query_dim = query_dim
        self.key_dim = key_dim if key_dim is not None else query_dim
        self.value_dim = value_dim if value_dim is not None else self.key_dim
        self.embed_dim = embed_dim if embed_dim is not None else query_dim

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.use_causal_mask = use_causal_mask
        self.scale = math.sqrt(self.head_dim)

        # Separate projections for Q, K, V (can have different input dimensions)
        self.query_projection = nn.Linear(self.query_dim, self.embed_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, self.embed_dim, bias=False)
        self.value_projection = nn.Linear(self.value_dim, self.embed_dim, bias=False)

        # Output projection (project back to query dimension)
        self.output_projection = nn.Linear(self.embed_dim, query_dim, bias=False)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.causal_mask_cache = None

    def _get_causal_mask(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask: query position i can only attend to key positions j where j <= i

        Args:
            query_len: Length of query sequence
            key_len: Length of key sequence
            device: Device

        Returns:
            mask: [query_len, key_len] where True = masked out
        """

        # Create upper triangular matrix
        mask = torch.triu(
            torch.ones(query_len, key_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        self.causal_mask_cache = mask
        return mask

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with separate query, key, and value sources.

        Args:
            query: [B, N_q, D_q] - query tensor
            key: [B, N_k, D_k] - key tensor. If None, uses query
            value: [B, N_v, D_v] - value tensor. If None, uses key

        Returns:
            output: [B, N_q, D_q] - attended output
        """
        batch_size = query.size(0)
        query_len = query.size(1)

        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = key

        key_len = key.size(1)
        value_len = value.size(1)

        assert key_len == value_len, "Key and value must have same sequence length"

        # Project to Q, K, V
        Q = self.query_projection(query)   # [B, N_q, embed_dim]
        K = self.key_projection(key)       # [B, N_k, embed_dim]
        V = self.value_projection(value)   # [B, N_v, embed_dim]

        # Reshape for multi-head: [B, N, embed_dim] -> [B, N, num_heads, head_dim]
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, value_len, self.num_heads, self.head_dim)

        # Transpose: [B, num_heads, N, head_dim]
        Q = Q.transpose(1, 2)  # [B, h, N_q, d]
        K = K.transpose(1, 2)  # [B, h, N_k, d]
        V = V.transpose(1, 2)  # [B, h, N_v, d]

        # Attention scores: [B, h, N_q, d] @ [B, h, d, N_k] -> [B, h, N_q, N_k]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply causal mask if enabled
        if self.use_causal_mask:
            causal_mask = self._get_causal_mask(query_len, key_len, query.device)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_q, N_k]
            attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [B, h, N_q, N_k]
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply to values: [B, h, N_q, N_k] @ [B, h, N_v, d] -> [B, h, N_q, d]
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads: [B, h, N_q, d] -> [B, N_q, h*d]
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, query_len, self.embed_dim)

        # Output projection
        output = self.output_projection(attention_output)
        output = self.output_dropout(output)

        return output  # [B, N_q, D_q]


class TransformerBlock(nn.Module):
    """
    Single transformer layer: Attention → Norm → MLP → Norm
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 use_self_attention: bool = True,
                 use_cross_attention: bool = False,
                 cross_attention_key_dim: Optional[int] = None,
                 cross_attention_value_dim: Optional[int] = None,
                 use_causal_mask: bool = False):
        """
        Args:
            dim: Dimension of input features
            num_heads: Number of attention heads
            mlp_ratio: Expansion ratio for MLP
            dropout: Dropout probability
            use_self_attention: If True, uses self-attention (mutually exclusive with cross-attention)
            use_cross_attention: If True, uses cross-attention (mutually exclusive with self-attention)
            cross_attention_key_dim: Dimension of cross-attention keys. If None, uses dim
            cross_attention_value_dim: Dimension of cross-attention values. If None, uses key_dim
            use_causal_mask: If True, apply causal masking in self-attention
        """
        super().__init__()

        if use_self_attention and use_cross_attention:
            raise ValueError("Cannot use both self-attention and cross-attention in single TransformerBlock")
        if not use_self_attention and not use_cross_attention:
            raise ValueError("Must use either self-attention or cross-attention")

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention

        if use_self_attention:
            self.attention = MultiHeadAttention(
                query_dim=dim,
                key_dim=None,
                value_dim=None,
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_causal_mask=use_causal_mask
            )
        else:
            self.attention = MultiHeadAttention(
                query_dim=dim,
                key_dim=cross_attention_key_dim,
                value_dim=cross_attention_value_dim,
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_causal_mask=False
            )

        self.norm1 = nn.LayerNorm(dim)

        mlp_hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self,
                x: torch.Tensor,
                context_key: Optional[torch.Tensor] = None,
                context_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: Attention → Norm → MLP → Norm

        Args:
            x: [B, N, D] - input tensor
            context_key: [B, M_k, D_k] - context for keys (only for cross-attention)
            context_value: [B, M_v, D_v] - context for values (only for cross-attention)

        Returns:
            output: [B, N, D] - transformed tensor
        """
        if self.use_self_attention:
            attn_out = self.attention(query=x)
        else:
            if context_key is None:
                raise ValueError("context_key required for cross-attention")
            if context_value is None:
                context_value = context_key
            attn_out = self.attention(query=x, key=context_key, value=context_value)

        x = x + attn_out
        x = self.norm1(x)

        mlp_out = self.mlp(x)

        x = x + mlp_out
        x = self.norm2(x)

        return x


class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them using a convolution
    Input: [B, C, H, W]
    Output: [B, num_patches, embed_dim]
    """
    def __init__(self, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(x)
        # [B, embed_dim, num_patches]
        x = x.flatten(2)
        # [B, num_patches, embed_dim]
        x = x.transpose(1, 2)

        x = self.norm(x)
        return x


class PatchUpsample(nn.Module):
    """
    Reconstruct image from patch embeddings using transposed convolution
    Input: [B, num_patches, embed_dim]
    Output: [B, out_channels, H, W]
    """
    def __init__(self, patch_size=4, embed_dim=384, out_channels=3, upsample=True):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        if upsample:
            self.proj = nn.ConvTranspose2d(
                embed_dim, out_channels,
                kernel_size=patch_size,
                stride=patch_size
            )
        else:
            self.proj = nn.Conv2d(
                embed_dim, out_channels,
                kernel_size=1,
                stride=1
            )

    def forward(self, x):
        # [B, num_patches, embed_dim]
        B, num_patches, embed_dim = x.shape
        h = w = int(num_patches ** 0.5)  # patches are squares of pixels
        # [B, embed_dim, h, w]
        x = x.transpose(1, 2)  # [B, embed_dim, num_patches]
        x = x.view(B, embed_dim, h, w)  # [B, embed_dim, h, w]
        # [B, out_channels, h*patch_size, w*patch_size]
        x = self.proj(x)
        return x


class TransformerNet(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 output_dim: Optional[int] = None,
                 only_use_encoder: bool = False,
                 use_decoder_masking: bool = True,
                 use_layerwise_connections: bool = False):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim is not None else dim
        self._only_use_encoder = only_use_encoder
        self.use_layerwise_connections = use_layerwise_connections

        # ===== ENCODER: N layers of self-attention =====
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_self_attention=True,
                use_cross_attention=False,
                use_causal_mask=False
            )
            for _ in range(num_layers)
        ])

        if not only_use_encoder:
            # ===== DECODER: N custom layers =====
            self.decoder_layers = nn.ModuleList()
            for _ in range(num_layers):
                decoder_layer = nn.ModuleDict({
                    'self_attention': MultiHeadAttention(
                        query_dim=dim,
                        key_dim=None,
                        value_dim=None,
                        embed_dim=dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_causal_mask=use_decoder_masking
                    ),
                    'norm1': nn.LayerNorm(dim),
                    'cross_attention': MultiHeadAttention(
                        query_dim=dim,
                        key_dim=dim,
                        value_dim=dim,
                        embed_dim=dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_causal_mask=False
                    ),
                    'norm2': nn.LayerNorm(dim),
                    'mlp': nn.Sequential(
                        nn.Linear(dim, dim * mlp_ratio),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim * mlp_ratio, dim),
                        nn.Dropout(dropout)
                    ),
                    'norm3': nn.LayerNorm(dim)
                })
                self.decoder_layers.append(decoder_layer)

        self.output_projection = nn.Linear(dim, self.output_dim)

    def forward(self,
                encoder_input: torch.Tensor,
                decoder_input: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_layerwise_connections and not self._only_use_encoder:
            encoder_outputs = []
            x = encoder_input
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                encoder_outputs.append(x)
            final_encoder_output = encoder_outputs[-1]
        else:
            encoder_output = encoder_input
            for encoder_layer in self.encoder_layers:
                encoder_output = encoder_layer(encoder_output)
            final_encoder_output = encoder_output

        if self._only_use_encoder:
            decoder_output = final_encoder_output
        else:
            if decoder_input is not None:
                decoder_output = decoder_input
            else:
                raise ValueError("Decoder input is None")

            for i, decoder_layer in enumerate(self.decoder_layers):
                self_attn_out = decoder_layer['self_attention'](query=decoder_output)
                decoder_output = decoder_output + self_attn_out
                decoder_output = decoder_layer['norm1'](decoder_output)

                if self.use_layerwise_connections:
                    encoder_feature = encoder_outputs[i]
                else:
                    encoder_feature = final_encoder_output

                cross_attn_out = decoder_layer['cross_attention'](
                    query=decoder_output,
                    key=encoder_feature,
                    value=encoder_feature
                )
                decoder_output = decoder_output + cross_attn_out
                decoder_output = decoder_layer['norm2'](decoder_output)

                mlp_out = decoder_layer['mlp'](decoder_output)
                decoder_output = decoder_output + mlp_out
                decoder_output = decoder_layer['norm3'](decoder_output)

        output = self.output_projection(decoder_output)
        return output


class ColorizationTransformerNet(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 num_color_tokens: int = 4096,
                 num_image_patches: int = 4096,
                 image_size: int = 256,
                 use_decoder_masking: bool = False,
                 only_use_encoder: bool = True,
                 output_channels: int = 3,
                 scale_embeddings: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.image_patches = num_image_patches
        self.output_channels = output_channels
        self.only_use_encoder = only_use_encoder
        self.num_color_tokens = num_color_tokens
        self.image_size = image_size
        # sqrt(d_model) embedding scaling from Attention Is All You Need, section 3.4
        self.embedding_scale = math.sqrt(embed_dim) if scale_embeddings else 1.0
        self.embedding_dropout = nn.Dropout(dropout)

        patch_size = image_size // int(num_image_patches ** 0.5)
        self.grayscale_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim
        )

        if not only_use_encoder:
            self.color_embedding = nn.Embedding(num_color_tokens, embed_dim)
            nn.init.normal_(self.color_embedding.weight, std=0.02)

            color_patch_size = image_size // int(num_color_tokens ** 0.5)
            self.color_upsample = PatchUpsample(
                patch_size=color_patch_size,
                embed_dim=embed_dim,
                out_channels=output_channels
            )
        else:
            self.patch_upsample = PatchUpsample(
                patch_size=patch_size,
                embed_dim=embed_dim,
                out_channels=output_channels
            )

        self.transformer = TransformerNet(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_layers=num_layers,
            output_dim=embed_dim,
            use_decoder_masking=use_decoder_masking,
            only_use_encoder=only_use_encoder,
            use_layerwise_connections=False
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        batch_size, channels, h, w = img.shape

        if channels == 3:
            grayscale_img = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            grayscale_img = img

        patch_size = self.image_size // int(self.image_patches ** 0.5)
        color_patch_size = self.image_size // int(self.num_color_tokens ** 0.5)
        if h % patch_size != 0 or w % patch_size != 0 or h % color_patch_size != 0 or w % color_patch_size != 0:
            raise ValueError(f"Image size ({h}x{w}) must be divisible by patch_size ({patch_size}) and color_patch_size ({color_patch_size})")

        grayscale_patches = self.grayscale_embed(grayscale_img)  # [B, num_patches, embed_dim]
        patch_h = h // patch_size
        patch_w = w // patch_size
        grayscale_pos_encoding = sinusoidalPositionalEncoding2D(patch_h, patch_w, self.embed_dim, grayscale_img.device)
        encoder_input = self.embedding_dropout(grayscale_patches * self.embedding_scale + grayscale_pos_encoding)

        if self.only_use_encoder:
            output_embeddings = self.transformer(
                encoder_input=encoder_input,
                decoder_input=None
            )  # [B, num_patches, embed_dim]
            output = self.patch_upsample(output_embeddings)  # [B, output_channels, H, W]
        else:
            color_indices = torch.arange(self.num_color_tokens, device=img.device)
            color_embeddings = self.color_embedding(color_indices)  # [num_color_tokens, embed_dim]
            color_patch_h = h // color_patch_size
            color_patch_w = w // color_patch_size
            color_pos = sinusoidalPositionalEncoding2D(color_patch_h, color_patch_w, self.embed_dim, grayscale_img.device)  # [num_color_tokens, embed_dim]

            decoder_input = color_embeddings.unsqueeze(0).repeat(batch_size, 1, 1) * self.embedding_scale + color_pos
            decoder_input = self.embedding_dropout(decoder_input)  # [B, num_color_tokens, embed_dim]

            output_embeddings = self.transformer(
                encoder_input=encoder_input,
                decoder_input=decoder_input
            )  # [B, num_color_tokens, embed_dim]

            output = self.color_upsample(output_embeddings)  # [B, output_channels, H, W]

        return output


class NeuralNetworkLayer(nn.Module):
    """
    A wrapper around a single torch.nn layer.
    Expects a dictionary of the form:
    {
        "type": "Linear", (for example, in general type is the name of the attribute,
                           representing a layer from torch.nn - str here)
        "params": {"in_features": 100, "out_features": 256} (for example, in general params is a dict[str:Any])
    }
    """

    def __init__(self, layer_config: Dict[str, Any]):
        super().__init__()
        layer_type = layer_config.get("type")
        params = layer_config.get("params", {})

        custom = False
        if hasattr(sys.modules[__name__], layer_type):
            custom = True
        elif not hasattr(nn, layer_type):
            raise UnsupportedLayerType(
                f"Unsupported layer type: {layer_type}. Not found in torch.nn or {__name__}."
            )
        try:
            if custom:
                self._layer = getattr(sys.modules[__name__], layer_type)(**params)
            else:
                self._layer = getattr(nn, layer_type)(**params)
        except TypeError as e:
            msg = str(e)
            unsupported_param = None
            if "unexpected keyword argument" in msg:
                unsupported_param = msg.split("'")[1]
                raise NotSupportedLayerConstructorParam(layer_type, unsupported_param) from e
            else:
                raise e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layer(x)


class NeuralNetwork(nn.Module):
    """
    Neural Network composed of sequential Layer modules.
    """

    def __init__(self, layers_config: List[Dict[str, Any]]):
        super().__init__()
        self._layers = nn.Sequential(*[NeuralNetworkLayer(layer_config) for layer_config in layers_config])

    def _validate(self, layers_list: List[nn.Module], index: int) -> bool:
        if not layers_list or index < 0 or index > len(layers_list) - 1:
            return False
        return True

    def _conditionalAccesToLayersList(self, index: int) -> List[nn.Module]:
        layers_list = list(self._layers.children())
        if self._validate(layers_list, index):
            return layers_list
        return []

    def insertLayer(self, layer: nn.Module, index: int = 0):
        layers_list = self._conditionalAccesToLayersList(index)
        if not layers_list:
            return
        layers_list.insert(index, layer)
        self._layers = nn.Sequential(*layers_list)

    def removeLayer(self, index: int = 0):
        layers_list = self._conditionalAccesToLayersList(index)
        if not layers_list:
            return
        del layers_list[index]
        self._layers = nn.Sequential(*layers_list)

    def applyDecoratorToLayerForwardMethod(self, decorator: Callable, index: int = 0):
        """
        Applies a decorator to the forward method of a given layer.
        """
        layers_list = self._conditionalAccesToLayersList(index)
        layers_list[index].forward = decorator(layers_list[index].forward).__get__(layers_list[index],
                                                                                   layers_list[index].__class__)
        self._layers = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


def createLayersFromConfig(architecture: List[Dict], custom_module: Optional[nn.Module] = None) -> nn.Module:
    layers = []
    for layer_config in architecture:
        layer_type = layer_config["type"]
        params = layer_config.get("params", {})

        if custom_module and hasattr(custom_module, layer_type):
            layer_class = getattr(custom_module, layer_type)
        elif custom_module and layer_type in globals():
            layer_class = globals()[layer_type]
        else:
            layer_class = getattr(nn, layer_type)

        layers.append(layer_class(**params))
    return nn.Sequential(*layers)


# torchvision's convnext_tiny.features interleaves downsampling layers with block stacks; these
# are the indices whose output ends a stage, i.e. the 1/4, 1/8, 1/16 and 1/32 feature maps.
CONVNEXT_TINY_STAGE_INDICES: Tuple[int, ...] = (1, 3, 5, 7)
CONVNEXT_TINY_STAGE_CHANNELS: Tuple[int, ...] = (96, 192, 384, 768)
# Every variant lays its features out identically, so the stage indices above hold for all of
# them; only the widths differ.
CONVNEXT_VARIANTS: Dict[str, Tuple[Any, Any, Tuple[int, ...]]] = {
    "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights, (96, 192, 384, 768)),
    "small": (convnext_small, ConvNeXt_Small_Weights, (96, 192, 384, 768)),
    "base": (convnext_base, ConvNeXt_Base_Weights, (128, 256, 512, 1024)),
}
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class FrozenConvNextBackbone(nn.Module):
    """ImageNet-pretrained ConvNeXt-Tiny run on luminance, frozen, returning its four stages.

    Supplies the semantic knowledge both colorization models lack. Measured on this dataset,
    the residual error of rgb_merge_unet_v5 and color_memory_transformer v6 is neither
    desaturation (predicted/target within-image ab std ratio 1.007) nor luminance error (mean
    |dL*| = 0.99) but hue assignment -- the models do not know what the regions they are
    colouring are. A backbone trained on far more images than this corpus contains is the
    cheapest source of that knowledge, and being frozen it also removes rather than adds
    degrees of freedom, which is the right direction for a model whose clean-train LPIPS
    (0.0975) already sits well below its validation LPIPS (0.1357).
    """

    def __init__(self, pretrained: bool = True, variant: str = "tiny") -> None:
        """variant: "tiny", "small" or "base".

        Tiny and small share the stage widths and differ only in depth, so swapping between
        them costs nothing downstream; base is wider and changes every lateral projection's
        input size. All three are frozen, so the extra capacity costs forward time and no
        gradient memory.
        """
        super().__init__()
        if variant not in CONVNEXT_VARIANTS:
            raise ValueError(f"unsupported ConvNeXt variant {variant!r}, expected one of "
                             f"{sorted(CONVNEXT_VARIANTS)}")
        builder, weights_enum, channels = CONVNEXT_VARIANTS[variant]
        self._variant = variant
        self._stage_channels = channels
        self._features = builder(weights=weights_enum.IMAGENET1K_V1 if pretrained else None
                                 ).features
        for parameter in self._features.parameters():
            parameter.requires_grad = False
        self._features.eval()

        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "FrozenConvNextBackbone":
        """Keep the backbone in eval mode whatever the parent does.

        ConvNeXt carries stochastic depth. Left in train mode it would randomly
        drop blocks of a network that is supposed to be a fixed feature extractor, making its
        output non-deterministic and different between the train and validation passes.
        """
        super().train(mode)
        self._features.eval()
        return self

    def getStageChannels(self) -> List[int]:
        return list(self._stage_channels)

    def forward(self, luminance: torch.Tensor) -> List[torch.Tensor]:
        """luminance: [B, 1, H, W] in [0, 1]. Returns the 1/4, 1/8, 1/16 and 1/32 feature maps."""
        if luminance.shape[1] != 1:
            raise ValueError(
                f"FrozenConvNextBackbone expects a single luminance channel, got "
                f"{luminance.shape[1]}"
            )

        normalized = (luminance.repeat(1, 3, 1, 1) - self._mean) / self._std

        stage_outputs: List[torch.Tensor] = []
        current = normalized
        # no_grad, not just requires_grad=False: the activations of a frozen branch never need
        # to be kept for backward, which is what keeps the memory cost at ~0.7 GiB at batch 16.
        with torch.no_grad():
            for index, stage in enumerate(self._features):
                current = stage(current)
                if index in CONVNEXT_TINY_STAGE_INDICES:
                    stage_outputs.append(current)
        return stage_outputs


class PretrainedSemanticEncoder(nn.Module):
    """Fuse a frozen backbone's feature pyramid into one map, for injection into a UNet.

    Produced at a single resolution (`output_stride`, default 8) because that is where the
    correction pays. Splitting rgb_merge_unet_v5's predicted chroma into a base band at 1/8
    resolution and the detail above it and crossing the bands with its own frozen extractors'
    output showed 56% of what the merge network buys lives in the base band and 7% in the
    detail band, so a second, finer injection point would target the 7%.

    The backbone is frozen; the lateral projections and the fusing convolution are trained.
    """

    def __init__(self, out_channels: int = 128, output_stride: int = 8,
                 norm_groups: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        if out_channels % norm_groups != 0:
            raise ValueError(
                f"norm_groups ({norm_groups}) must divide out_channels ({out_channels})"
            )

        self.out_channels = out_channels
        self.output_stride = output_stride
        self.backbone = FrozenConvNextBackbone(pretrained=pretrained)

        self.lateral_projections = nn.ModuleList([
            nn.Conv2d(stage_channels, out_channels, kernel_size=1)
            for stage_channels in self.backbone.getStageChannels()
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU()
        )

    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        """luminance: [B, 1, H, W]. Returns [B, out_channels, H / stride, W / stride]."""
        stage_outputs = self.backbone(luminance)

        height = luminance.shape[-2] // self.output_stride
        width = luminance.shape[-1] // self.output_stride

        fused: Optional[torch.Tensor] = None
        for projection, stage_output in zip(self.lateral_projections, stage_outputs):
            lateral = projection(stage_output.float())
            if lateral.shape[-2:] != (height, width):
                lateral = F.interpolate(lateral, size=(height, width),
                                        mode="bilinear", align_corners=False)
            fused = lateral if fused is None else fused + lateral

        return self.fuse(fused)


class FilmModulation(nn.Module):
    """Condition a feature map on a per-image vector by scaling and shifting its channels.

    ColorMemoryTransformer can take the cluster prior as a bias on its colour queries, because
    it has a query set and the prior is a statement about the image as a whole. A UNet has no
    such place: its tensors are spatial and every position would have to receive the same
    vector, which is what concatenating a broadcast vector amounts to -- at the cost of widening
    every downstream convolution. Feature-wise modulation says the same thing more cheaply: the
    prior decides which channels of the bottleneck matter and by how much.

    Initialised to the identity (zero scale, zero shift) so that a freshly built model starts
    exactly where the unconditioned one does and has to learn to use the prior.
    """

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.to_modulation = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.to_modulation.weight)
        nn.init.zeros_(self.to_modulation.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        modulation = self.to_modulation(condition)
        scale, shift = modulation.chunk(2, dim=1)
        return features * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class MultiScaleSemanticEncoder(nn.Module):
    """A frozen backbone's pyramid delivered at several resolutions instead of fused into one.

    PretrainedSemanticEncoder answers "what is here" once, at stride 8, and the decoder carries
    that answer from memory through every later stage. Measured on this dataset that single
    injection bought rgb_merge_unet_v6 7% of raw LPIPS, while replacing colour_memory_transformer's
    whole encoder with the same backbone -- semantics available at every scale the decoder works
    at -- bought 18%. This closes that gap for the UNet: each decoder level gets the pyramid
    level at its own resolution, so a block refining 1/4-resolution detail is told what the
    region is at 1/4 resolution rather than inferring it from a 1/8 summary.

    Built top-down (FPN): each level is its own lateral projection plus the coarser level
    upsampled, so fine levels inherit the coarse levels' larger receptive field instead of
    seeing only the backbone's early, semantically weak features.
    """

    STAGE_STRIDES: Tuple[int, ...] = (4, 8, 16, 32)
    # The backbone stops at 1/4, but a UNet decoder keeps going to 1/2 and 1/1. Those levels are
    # served by upsampling the finest backbone level: it adds no new information, it lets a
    # block refining full-resolution detail condition on what the region is. They are expensive
    # -- a 1/1 map at 128 channels and batch 16 is ~0.5 GiB of activations -- so they are opt-in.
    UPSAMPLED_STRIDES: Tuple[int, ...] = (1, 2)

    def __init__(self, out_channels: int = 128, output_strides: Optional[List[int]] = None,
                 norm_groups: int = 8, pretrained: bool = True,
                 variant: str = "tiny") -> None:
        super().__init__()
        if out_channels % norm_groups != 0:
            raise ValueError(
                f"norm_groups ({norm_groups}) must divide out_channels ({out_channels})"
            )
        output_strides = list(output_strides) if output_strides is not None else [4, 8, 16]
        allowed = set(self.STAGE_STRIDES) | set(self.UPSAMPLED_STRIDES)
        unknown = set(output_strides) - allowed
        if unknown:
            raise ValueError(f"output_strides {sorted(unknown)} are neither backbone strides "
                             f"{list(self.STAGE_STRIDES)} nor upsampled ones "
                             f"{list(self.UPSAMPLED_STRIDES)}")

        self.out_channels = out_channels
        self.output_strides = sorted(output_strides)
        self.backbone = FrozenConvNextBackbone(pretrained=pretrained, variant=variant)

        self.lateral_projections = nn.ModuleList([
            nn.Conv2d(stage_channels, out_channels, kernel_size=1)
            for stage_channels in self.backbone.getStageChannels()
        ])
        # One fusing conv per emitted level; levels that are only passed through on the way
        # down do not need one, so this stays proportional to what is actually consumed.
        self.fusions = nn.ModuleDict({
            str(stride): nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(norm_groups, out_channels),
                nn.GELU()
            )
            for stride in self.output_strides
        })

    def computeStages(self, luminance: torch.Tensor) -> List[torch.Tensor]:
        """The frozen backbone's raw pyramid, exposed so one forward can serve two consumers.

        The cluster prior reads the same features. Running the backbone twice would double the
        one genuinely expensive part of this branch for no gain.
        """
        return self.backbone(luminance)

    def fusePyramid(self, stage_outputs: List[torch.Tensor]) -> Dict[int, torch.Tensor]:
        laterals = [projection(stage_output.float())
                    for projection, stage_output in zip(self.lateral_projections, stage_outputs)]

        current = laterals[-1]
        by_stride: Dict[int, torch.Tensor] = {self.STAGE_STRIDES[-1]: current}
        for index in range(len(laterals) - 2, -1, -1):
            current = laterals[index] + F.interpolate(
                current, size=laterals[index].shape[-2:], mode="bilinear", align_corners=False
            )
            by_stride[self.STAGE_STRIDES[index]] = current

        finest = by_stride[self.STAGE_STRIDES[0]]
        for stride in self.UPSAMPLED_STRIDES:
            if stride in self.output_strides:
                scale = self.STAGE_STRIDES[0] // stride
                by_stride[stride] = F.interpolate(
                    finest, scale_factor=scale, mode="bilinear", align_corners=False
                )

        return {stride: self.fusions[str(stride)](by_stride[stride])
                for stride in self.output_strides}

    def forward(self, luminance: torch.Tensor) -> Dict[int, torch.Tensor]:
        """luminance: [B, 1, H, W]. Returns {stride: [B, out_channels, H/stride, W/stride]}."""
        return self.fusePyramid(self.computeStages(luminance))


class PretrainedLuminanceEncoder(nn.Module):
    """Drop-in replacement for LuminanceEncoder backed by a frozen ImageNet backbone.

    ConvNeXt-Tiny's stage pyramid has exactly the geometry ColorMemoryTransformer's
    PixelDecoder already expects from luminance_encoder_v3 -- four levels, 2x apart, starting
    at 1/4 -- so nothing downstream needs modifying: PixelDecoder sizes itself from
    getDownsampleChannels().

    `stage_channels` exists because the widths do differ ([96, 192, 384, 768] against the
    from-scratch encoder's [96, 256, 256, 512]) and PixelDecoder's parameter count follows
    them: handing it the raw ConvNeXt widths grows the decoder by ~10M parameters, which would
    confound "semantics from ImageNet instead of from scratch" with "a bigger decoder" in any
    comparison against the previous version. Passing the old widths inserts a trained 1x1
    projection per stage and leaves the rest of the network parameter-identical. None passes
    the backbone's own widths straight through.
    """

    def __init__(self, pretrained: bool = True,
                 stage_channels: Optional[List[int]] = None,
                 variant: str = "tiny") -> None:
        super().__init__()
        self.backbone = FrozenConvNextBackbone(pretrained=pretrained, variant=variant)
        backbone_channels = self.backbone.getStageChannels()

        if stage_channels is not None and len(stage_channels) != len(backbone_channels):
            raise ValueError(
                f"stage_channels must have one entry per backbone stage "
                f"({len(backbone_channels)}), got {len(stage_channels)}"
            )

        self._stage_channels: List[int] = (
            list(stage_channels) if stage_channels is not None else backbone_channels
        )
        self.stage_projections: Optional[nn.ModuleList] = None
        if stage_channels is not None:
            self.stage_projections = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels else nn.Identity()
                for in_channels, out_channels in zip(backbone_channels, self._stage_channels)
            ])

    def getDownsampleChannels(self) -> List[int]:
        return list(self._stage_channels)

    def getNumDownsampleLayers(self) -> int:
        return len(self._stage_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        stage_outputs = [stage_output.float() for stage_output in self.backbone(x)]
        if self.stage_projections is None:
            return stage_outputs
        return [projection(stage_output)
                for projection, stage_output in zip(self.stage_projections, stage_outputs)]


class LuminanceEncoder(nn.Module):
    def __init__(self, encoder_module: Optional[nn.Module] = None, config_path: Optional[str] = None):
        super().__init__()
        self.feature_outputs = []
        self.downsample_channels = []

        if encoder_module is not None:
            self.layers = encoder_module
            self._analyzeChannelsFromModule(encoder_module)
        elif config_path is not None:
            self.layers = self.buildEncoderFromConfig(config_path)
            self._analyzeChannelsFromConfig(config_path)
        else:
            self.layers = self.buildDefaultEncoder()
            self.downsample_channels = [64, 128, 256, 512]

    def _analyzeChannelsFromModule(self, module: nn.Module):
        for layer in module:
            if isinstance(layer, nn.Conv2d) and hasattr(layer, 'stride'):
                if (isinstance(layer.stride, int) and layer.stride > 1) or \
                   (isinstance(layer.stride, tuple) and any(s > 1 for s in layer.stride)):
                    self.downsample_channels.append(layer.out_channels)

    def _analyzeChannelsFromConfig(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                architecture = json.load(f)

            for layer_config in architecture:
                layer_type = layer_config["type"]
                params = layer_config.get("params", {})

                if layer_type == "Conv2d" and params.get("stride", 1) > 1:
                    self.downsample_channels.append(params["out_channels"])
        except Exception as e:
            raise RuntimeError(f"Failed to analyze channels from config {config_path}: {e}")

    def buildDefaultEncoder(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def buildEncoderFromConfig(self, config_path: str) -> nn.Module:
        try:
            with open(config_path, 'r') as f:
                architecture = json.load(f)
            return createLayersFromConfig(architecture, self)
        except Exception as e:
            raise RuntimeError(f"Failed to load encoder from {config_path}: {e}")

    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], torch.Tensor]:
        self.feature_outputs.clear()
        current = x

        if not isinstance(self.layers, nn.Sequential):
            if hasattr(self.layers, 'forward') and self.layers.forward.__code__.co_argcount == 1:
                result = self.layers(x)
                if result and isinstance(result, list):
                    return result
            raise ValueError("Custom encoder module must return feature_outputs list.")

        for layer in self.layers:
            is_downsample = False
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)) and hasattr(layer, 'stride'):
                if isinstance(layer.stride, int):
                    is_downsample = layer.stride > 1
                elif isinstance(layer.stride, tuple):
                    is_downsample = any(stride > 1 for stride in layer.stride)

            current = layer(current)

            if is_downsample:
                self.feature_outputs.append(current)

        return self.feature_outputs

    def getDownsampleChannels(self) -> List[int]:
        return self.downsample_channels

    def getNumDownsampleLayers(self) -> int:
        return len(self.downsample_channels)


class PixelDecoder(nn.Module):
    def __init__(self,
                 encoder_channels: List[int],
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = len(encoder_channels)

        self.transformer_blocks = nn.ModuleList()
        decoder_channels = list(reversed(encoder_channels))

        for i in range(self.num_layers):
            current_dim = decoder_channels[i]
            transformer_block = TransformerBlock(
                dim=current_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_self_attention=True,
                use_cross_attention=False,
                use_causal_mask=False
            )
            self.transformer_blocks.append(transformer_block)

        self.upsample_layers = nn.ModuleList()
        self.output_projections = nn.ModuleList()

        self.feature_embeddings = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = decoder_channels[i]
            if i < self.num_layers - 1:
                out_channels = decoder_channels[i+1]
            else:
                out_channels = self.embed_dim

            upsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),  # [B, out_channels*4, H, W]
                nn.PixelShuffle(2),  # [B, out_channels, H*2, W*2]
                nn.GELU()
            )
            self.upsample_layers.append(upsample_layer)

            feature_embed = nn.Embedding(1, out_channels)
            nn.init.zeros_(feature_embed.weight)
            self.feature_embeddings.append(feature_embed)

            if out_channels != self.embed_dim:
                self.output_projections.append(nn.Linear(out_channels, self.embed_dim))
            else:
                self.output_projections.append(nn.Identity())

    def forward(self, encoder_outputs: List[torch.Tensor]) -> tuple[List[torch.Tensor], torch.Tensor]:
        batch_size = encoder_outputs[0].shape[0]
        device = encoder_outputs[0].device

        pixel_decoder_outputs = []

        b, c, h_enc, w_enc = encoder_outputs[-1].shape
        current_features = encoder_outputs[-1].view(b, c, -1).transpose(1, 2)  # [batch_size, h_enc*w_enc, decoder_channels[0]]

        for i in range(self.num_layers):
            if i >= 1:
                residual_input = encoder_outputs[-(i+1)]  # [batch_size, decoder_channels[i], H_res, W_res]

                b_res, c_res, h_res, w_res = residual_input.shape
                residual_patches = residual_input.view(b_res, c_res, -1).transpose(1, 2)  # [batch_size, h_res*w_res, decoder_channels[i]]

                transformer_input = current_features + residual_patches
                h_res = residual_input.shape[2]
                w_res = residual_input.shape[3]
            else:
                transformer_input = current_features
                h_res = h_enc
                w_res = w_enc

            pos_encoding = sinusoidalPositionalEncoding2D(h_res, w_res, current_features.shape[-1], device)  # [1, h_res*w_res, decoder_channels[i]]
            transformer_input_with_pos = transformer_input + pos_encoding

            transformer_output = self.transformer_blocks[i](transformer_input_with_pos)  # [batch_size, h_res*w_res, decoder_channels[i]]

            b, n_patches, c = transformer_output.shape
            spatial_features = transformer_output.transpose(1, 2).view(b, c, h_res, w_res)  # [batch_size, decoder_channels[i], h_res, w_res]

            upsampled_features = self.upsample_layers[i](spatial_features)  # [batch_size, out_channels, h_up, w_up]

            b, c_up, h_up, w_up = upsampled_features.shape

            output_features = upsampled_features.view(b, c_up, -1).transpose(1, 2)  # [batch_size, h_up*w_up, out_channels]

            feature_embed = self.feature_embeddings[i].weight.unsqueeze(0)  # [1, 1, out_channels]
            feature_embed = feature_embed.expand(batch_size, output_features.size(1), -1)  # [batch_size, h_up*w_up, out_channels]

            output_features_with_embed = output_features + feature_embed  # [batch_size, h_up*w_up, out_channels]

            output_features_projected = self.output_projections[i](output_features_with_embed)  # [batch_size, h_up*w_up, embed_dim]

            pos_encoding = sinusoidalPositionalEncoding2D(h_up, w_up, self.embed_dim, device)  # [1, h_up*w_up, D]
            output_features_projected_with_pos = output_features_projected + pos_encoding  # [batch_size, h_up*w_up, embed_dim]
            pixel_decoder_outputs.append(output_features_projected_with_pos)

            current_features = output_features_with_embed  # [batch_size, h_up*w_up, out_channels]

        b, n_patches, c_final = current_features.shape
        h_final = h_enc * (2 ** self.num_layers)
        w_final = w_enc * (2 ** self.num_layers)
        final_pixel_spatial = current_features.transpose(1, 2).view(b, c_final, h_final, w_final)  # [batch_size, c_final, h_final, w_final]

        return pixel_decoder_outputs, final_pixel_spatial


# Standard deviation for the color/memory query tables of MultiScaleColorDecoder. Kept at the
# usual transformer embedding scale: large enough to break the query symmetry immediately,
# small enough that the first decoder block still starts near its identity behaviour.
QUERY_EMBEDDING_INIT_STD: float = 0.02


CLUSTER_PRIOR_COLORS: Tuple[str, ...] = (
    "black", "blue", "brown", "cyan", "gray", "green", "magenta",
    "orange", "pink", "purple", "red", "white", "yellow"
)
CLUSTER_PRIOR_STATISTICS: Tuple[str, ...] = ("share", "saturation_mean", "value_mean")


class ClusterColorPrior(nn.Module):
    """Which colours images of this kind actually have, told to the model from its own weights.

    The residual error of these models is hue assignment on subject matter whose colour is not
    determined by its shape: measured on the validation tail, chroma magnitude is right (ratio
    0.875) while the chroma-weighted hue error is 61 degrees, and the model answers ambiguity by
    collapsing onto one hue (circular concentration 0.670 against the ground truth's 0.518).
    A poster's colour cannot be read off a grey poster -- but the distribution of colours that
    posters have can be learnt, and it is already measured, in Phase 0's cluster_color_sv.csv.

    Everything needed at inference is a buffer: the frozen linear head that names the cluster
    from the frozen backbone's pooled features, the feature normalisation it expects, and the
    per-cluster colour table. They all travel inside state_dict, so a deployed checkpoint needs
    no CSV, no cluster assignment and no access to the corpus -- the input stays luminance
    alone. Buffers start at zero rather than being read from disk in __init__ for the same
    reason: constructing the model must not require the build-time files.

    The cluster is predicted, not looked up, so it is sometimes wrong: a linear probe on frozen
    features reaches 62% top-1 and 90% top-5 over 98 clusters, and no worse on the tail clusters
    this is meant to help. The mixture is therefore soft -- the prior is the posterior-weighted
    average of the table's rows, not the row of the argmax -- so a confident-but-wrong cluster
    degrades the hint rather than replacing it with a wrong one.
    """

    def __init__(self, feature_dim: int, output_dim: int, num_clusters: int = 98,
                 temperature: float = 1.0, tables_path: Optional[str] = None,
                 backbone_variant: Optional[str] = None) -> None:
        """tables_path: a .pt written by analysis/build_cluster_prior.py, or None when the
        tables will arrive from a checkpoint. Either way the values end up in buffers; forward
        refuses to run while they are still zero rather than silently conditioning on nothing.
        """
        super().__init__()
        prior_dim = len(CLUSTER_PRIOR_COLORS) * len(CLUSTER_PRIOR_STATISTICS)
        self.num_clusters = num_clusters
        self.prior_dim = prior_dim
        self.temperature = temperature
        self.backbone_variant = backbone_variant

        self.register_buffer("feature_mean", torch.zeros(1, feature_dim))
        self.register_buffer("feature_std", torch.ones(1, feature_dim))
        self.register_buffer("head_weight", torch.zeros(num_clusters, feature_dim))
        self.register_buffer("head_bias", torch.zeros(num_clusters))
        self.register_buffer("color_statistics", torch.zeros(num_clusters, prior_dim))

        self.projection = nn.Sequential(
            nn.Linear(prior_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

        if tables_path is not None:
            # Absent is tolerated, because the tables are also in every checkpoint this model
            # ever saves: at inference the architecture is built and then immediately overwritten
            # by load_state_dict, so demanding the build-time file there would make a deployment
            # depend on a file it does not use. Getting it genuinely wrong is still caught -- the
            # buffers stay zero and forward refuses to run.
            if os.path.exists(tables_path):
                self.loadTables(**torch.load(tables_path, map_location="cpu",
                                             weights_only=True))
            else:
                warnings.warn(
                    f"cluster prior tables {tables_path!r} not found; the buffers stay zero and "
                    f"must arrive from a checkpoint, or forward will refuse to run",
                    RuntimeWarning
                )

    def loadTables(self, feature_mean: torch.Tensor, feature_std: torch.Tensor,
                   head_weight: torch.Tensor, head_bias: torch.Tensor,
                   color_statistics: torch.Tensor, variant: Optional[str] = None) -> None:
        """Fill the frozen tables at build time; afterwards they live in the checkpoint.

        The variant check is not redundant with the shape check: ConvNeXt-Tiny and Small have
        identical stage widths, so a head trained on one loads into the other without a single
        mismatched dimension and then classifies features from a network it never saw. The
        symptom would be a prior that simply never helps.
        """
        if (variant is not None and self.backbone_variant is not None
                and variant != self.backbone_variant):
            raise ValueError(
                f"cluster prior tables were built on ConvNeXt-{variant} but this model runs "
                f"ConvNeXt-{self.backbone_variant}. Their stage widths may match, so nothing "
                f"else would catch this; rebuild the tables with --variant {self.backbone_variant}"
            )
        for name, value in (("feature_mean", feature_mean), ("feature_std", feature_std),
                            ("head_weight", head_weight), ("head_bias", head_bias),
                            ("color_statistics", color_statistics)):
            buffer = getattr(self, name)
            if value.shape != buffer.shape:
                raise ValueError(f"{name} must have shape {tuple(buffer.shape)}, "
                                 f"got {tuple(value.shape)}")
            buffer.copy_(value.to(buffer.dtype))

    def forward(self, backbone_features: List[torch.Tensor]) -> torch.Tensor:
        """backbone_features: the frozen pyramid. Returns [B, output_dim]."""
        if not bool(self.head_weight.any()):
            raise RuntimeError(
                "ClusterColorPrior has no tables: pass tables_path when building a fresh model, "
                "or load a checkpoint that carries them. Running with zeros would condition "
                "every image on the same constant and look like a model that simply ignores "
                "the prior."
            )

        pooled = torch.cat([stage.float().mean(dim=(2, 3)) for stage in backbone_features],
                           dim=1)
        normalized = (pooled - self.feature_mean) / self.feature_std
        logits = F.linear(normalized, self.head_weight, self.head_bias) / self.temperature
        posterior = logits.softmax(dim=1)
        return self.projection(posterior @ self.color_statistics)


class QuantizedChromaHead(nn.Module):
    """Predict a distribution over ab cells instead of two numbers, and decode it continuously.

    A regressor can only answer with one colour. Asked for a poster that is red in half the
    corpus and blue in the other half, the value that minimises its loss is the average, and the
    average of red and blue is a purple that is wrong for both. Measured on this validation set,
    73% of the tail images would score better under some rigid rotation of their predicted hue,
    chosen on pixels the score never saw -- the structure is right and the colour choice is not,
    which is the failure a distribution can express and a regressor cannot.

    Decoding is the annealed mean: softmax(logits / T) against the cell centres. It is
    continuous, so the output is not confined to the palette -- measured on ground-truth chroma,
    snapping to a 10-unit grid costs 0.0507 raw LPIPS while the annealed mean over the same grid
    costs 0.0047, an order of magnitude less than the gain being chased.

    Class rebalancing exists because the corpus is warm: one cell holds 15% of all pixels, so a
    head trained on raw frequencies learns that betting on sepia almost always pays. The weights
    follow Zhang et al., mixing the empirical distribution with a uniform one.
    """

    def __init__(self, in_channels: int, num_bins: int, temperature: float = 0.38,
                 rebalance_lambda: float = 0.5, encode_sigma: float = 5.0,
                 encode_neighbours: int = 5, bins_path: Optional[str] = None,
                 upsample_factor: int = 1, encode_chunk: int = 131072) -> None:
        super().__init__()
        if not 0.0 < temperature <= 1.0:
            raise ValueError(f"temperature must be in (0, 1], got {temperature}")
        if not 0.0 <= rebalance_lambda <= 1.0:
            raise ValueError(f"rebalance_lambda must be in [0, 1], got {rebalance_lambda}")

        self.num_bins = num_bins
        self.temperature = temperature
        # Logits are the expensive tensor: Q channels at full resolution and batch 16 is 0.9 GiB
        # each, and autograd keeps several. Predicting them one octave down and upsampling the
        # decoded ab costs almost nothing -- ground-truth chroma band-limited to 1/4 scores
        # 0.031 raw LPIPS against the model's 0.129 -- and divides the head's memory by four.
        self.upsample_factor = upsample_factor
        self.encode_chunk = encode_chunk
        self.rebalance_lambda = rebalance_lambda
        self.encode_sigma = encode_sigma
        self.encode_neighbours = min(encode_neighbours, num_bins)

        self.logits_conv = nn.Conv2d(in_channels, num_bins, kernel_size=3, padding=1)
        self.register_buffer("bin_centres", torch.zeros(num_bins, 2))
        self.register_buffer("bin_frequencies", torch.zeros(num_bins))
        self.register_buffer("class_weights", torch.ones(num_bins))

        if bins_path is not None:
            if os.path.exists(bins_path):
                self.loadBins(**torch.load(bins_path, map_location="cpu", weights_only=True))
            else:
                warnings.warn(
                    f"chroma bins {bins_path!r} not found; the palette stays zero and must "
                    f"arrive from a checkpoint, or forward will refuse to run",
                    RuntimeWarning
                )

    def loadBins(self, bin_centres: torch.Tensor, bin_frequencies: torch.Tensor,
                 bin_size: Optional[float] = None) -> None:
        if bin_centres.shape != (self.num_bins, 2):
            raise ValueError(f"bin_centres must have shape {(self.num_bins, 2)}, "
                             f"got {tuple(bin_centres.shape)}; num_bins in the architecture has "
                             f"to match the palette file")
        self.bin_centres.copy_(bin_centres)
        self.bin_frequencies.copy_(bin_frequencies)
        self.class_weights.copy_(self._computeClassWeights(bin_frequencies))

    def _computeClassWeights(self, frequencies: torch.Tensor) -> torch.Tensor:
        mixed = ((1.0 - self.rebalance_lambda) * frequencies
                 + self.rebalance_lambda / self.num_bins)
        weights = 1.0 / mixed
        # Normalised so the expected weight under the data is 1: rebalancing should change which
        # errors matter, not the overall scale of the loss.
        return weights / (frequencies * weights).sum().clamp_min(1e-8)

    def _assertLoaded(self) -> None:
        if not bool(self.bin_centres.any()):
            raise RuntimeError(
                "QuantizedChromaHead has no palette: pass bins_path when building a fresh model, "
                "or load a checkpoint that carries it. Decoding against zeros would return "
                "neutral for every pixel."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, C, H, W]. Returns ab in [B, 2, H, W], and keeps the logits for the loss."""
        self._assertLoaded()
        logits = self.logits_conv(features)
        self.last_logits = logits
        chroma = self.decode(logits)
        if self.upsample_factor > 1:
            chroma = F.interpolate(chroma, scale_factor=self.upsample_factor,
                                   mode="bilinear", align_corners=False)
        return chroma

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits / self.temperature, dim=1)
        return torch.einsum("bqhw,qc->bchw", probabilities, self.bin_centres)

    def softEncode(self, chroma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ground-truth ab as its nearest cells and their Gaussian weights, kept sparse.

        Returns (indices, weights), each [N, k] rather than a dense [N, num_bins] map: at
        256x256 and batch 16 the dense form is 0.9 GiB, and only k of its entries are non-zero.
        The distances are computed in chunks for the same reason -- the full [N, num_bins]
        distance matrix is the same size again.

        A one-hot target would make every near-miss as wrong as the opposite side of the wheel,
        which is exactly the distinction this head exists to keep.
        """
        self._assertLoaded()
        flat = chroma.permute(0, 2, 3, 1).reshape(-1, 2)

        index_chunks: List[torch.Tensor] = []
        weight_chunks: List[torch.Tensor] = []
        for start in range(0, flat.shape[0], self.encode_chunk):
            block = flat[start:start + self.encode_chunk]
            distances = torch.cdist(block, self.bin_centres)
            nearest = distances.topk(self.encode_neighbours, dim=1, largest=False)
            weights = torch.exp(-nearest.values.pow(2) / (2.0 * self.encode_sigma ** 2))
            index_chunks.append(nearest.indices)
            weight_chunks.append(weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8))

        return torch.cat(index_chunks), torch.cat(weight_chunks)

    def classificationLoss(self, logits: torch.Tensor, chroma: torch.Tensor) -> torch.Tensor:
        """Rebalanced cross entropy of the logits against the soft-encoded ground truth.

        The target is gathered rather than materialised: cross entropy against a distribution
        with k non-zero entries needs k terms, not num_bins of which all but k are zero.
        """
        if chroma.shape[-2:] != logits.shape[-2:]:
            # The head may predict one octave below the image; the target follows it down rather
            # than the logits being upsampled, which would defeat the point of predicting low.
            chroma = F.interpolate(chroma, size=logits.shape[-2:], mode="bilinear",
                                   align_corners=False)

        indices, weights = self.softEncode(chroma)
        log_probabilities = torch.log_softmax(logits, dim=1)
        flat_log_probabilities = log_probabilities.permute(0, 2, 3, 1).reshape(
            -1, self.num_bins
        )

        gathered = torch.gather(flat_log_probabilities, 1, indices)
        per_pixel = -(weights * gathered).sum(dim=1)

        dominant = indices.gather(1, weights.argmax(dim=1, keepdim=True)).squeeze(1)
        return (per_pixel * self.class_weights[dominant]).mean()


class MultiScaleColorDecoder(nn.Module):
    def __init__(self,
                 color_dim: int = 256,
                 embed_dim: int = 256,
                 output_dim: int = 256,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 memory_size: int = 256):
        """color_dim and embed_dim must be equal; output_dim is free.

        They read like independent widths and the first two are not. The queries are color_dim
        wide, the attention returns embed_dim, and the two are summed as a residual; the norms
        and the MLP are then built on embed_dim and applied to that same sum. A mismatch used to
        surface as a shape error at the first forward pass rather than at construction, which is
        what the old default pair (256 against 512) produced.

        output_dim only sizes the final projection, so this module is happy with any value; the
        constraint that ties it to embed_dim belongs to ColorMemoryTransformer, which contracts
        this output against its pixel features.
        """
        super().__init__()
        if color_dim != embed_dim:
            raise ValueError(
                f"color_dim ({color_dim}) and embed_dim ({embed_dim}) must be equal: the "
                f"attention output is summed into the queries as a residual, and the norms and "
                f"MLP are built on that same width"
            )

        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.num_layers = num_layers

        # Both tables must start off random rather than at zero. Every operation in this
        # decoder is applied query-wise, so identical rows stay identical the whole way
        # through the blocks; with zero init all `memory_size` queries emit the same
        # affinity map, and the only thing that can tell them apart is the per-channel
        # weights of the downstream smoothing head. Differentiating through that single
        # weak path leaves the memory collapsed -- measured on the v5 checkpoint, 128
        # queries spanned an effective rank of ~13. A small random init breaks the symmetry
        # at step zero, the way Mask2Former initialises its query embeddings.
        self.color_embeddings = nn.Embedding(memory_size, color_dim)
        nn.init.trunc_normal_(self.color_embeddings.weight, std=QUERY_EMBEDDING_INIT_STD)

        self.memory_embeddings = nn.Embedding(memory_size, color_dim)
        nn.init.trunc_normal_(self.memory_embeddings.weight, std=QUERY_EMBEDDING_INIT_STD)

        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            decoder_block = nn.ModuleDict({
                'cross_attention': MultiHeadAttention(
                    query_dim=color_dim,
                    key_dim=embed_dim,
                    value_dim=embed_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_causal_mask=False
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'self_attention': MultiHeadAttention(
                    query_dim=color_dim,
                    key_dim=None,
                    value_dim=None,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_causal_mask=False
                ),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * mlp_ratio),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * mlp_ratio, embed_dim),
                    nn.Dropout(dropout)
                ),
                'norm3': nn.LayerNorm(embed_dim)
            })
            self.decoder_blocks.append(decoder_block)
        self._final_projection = nn.Linear(embed_dim, output_dim)

    def forward(self, pixel_decoder_outputs: List[torch.Tensor],
                color_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = pixel_decoder_outputs[0].shape[0]

        color_queries = self.color_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, memory_size, color_dim]

        if color_prior is not None:
            # One bias per image, shared by every query: the prior says what colours this kind
            # of image has, which is a statement about the image and not about any one query.
            # Added rather than concatenated so the queries keep the width the decoder blocks
            # were built for, and so a zero prior leaves the model exactly as it was.
            color_queries = color_queries + color_prior.unsqueeze(1)

        memory = self.memory_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, memory_size, color_dim]

        color_decoder_output = color_queries

        for i, block in enumerate(self.decoder_blocks):
            pixel_features_idx = i % len(pixel_decoder_outputs)
            pixel_features = pixel_decoder_outputs[pixel_features_idx]  # [batch_size, num_patches, embed_dim]

            queries_with_pos = color_decoder_output + memory  # [batch_size, memory_size, color_dim]

            cross_attn_out = block['cross_attention'](
                query=queries_with_pos,
                key=pixel_features,
                value=pixel_features
            )  # [batch_size, memory_size, color_dim]

            color_decoder_output = color_decoder_output + cross_attn_out
            color_decoder_output = block['norm1'](color_decoder_output)

            self_attn_out = block['self_attention'](query=color_decoder_output + memory)  # [batch_size, memory_size, color_dim]
            color_decoder_output = color_decoder_output + self_attn_out
            color_decoder_output = block['norm2'](color_decoder_output)

            mlp_out = block['mlp'](color_decoder_output)  # [batch_size, memory_size, color_dim]
            color_decoder_output = color_decoder_output + mlp_out
            color_decoder_output = block['norm3'](color_decoder_output)

        return self._final_projection(color_decoder_output)  # [batch_size, memory_size, color_dim] -> [batch_size, memory_size, output_dim]


class ScaledTanh(nn.Module):
    """Bounded output activation computing ``scale * tanh(x)``.

    Intended as the final colorisation layer: it keeps the regressed AB channels inside
    Kornia's native Lab gamut (roughly [-scale, scale]) instead of letting them drift to
    extreme values, while staying smooth and differentiable. The default scale matches the
    AB range documented by the ExtractABChannels transform.
    """

    def __init__(self, scale: float = 127.0) -> None:
        super().__init__()
        self.scale: float = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(x)


class ColorMemoryTransformer(nn.Module):
    def __init__(self,
                 color_dim: int = 256,
                 embed_dim: int = 256,
                 color_decoder_output_dim: int = 256,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 color_decoder_layers: int = 6,
                 memory_size: int = 256,
                 smoothing_config_path: str = None,
                 encoder_module: Optional[nn.Module] = None,
                 encoder_config_path: str = None,
                 pretrained_encoder: Optional[str] = None,
                 pretrained_encoder_stage_channels: Optional[List[int]] = None,
                 cluster_color_prior: Optional[Dict[str, Any]] = None,
                 quantized_chroma_head: Optional[Dict[str, Any]] = None):
        """
        pretrained_encoder: name of a frozen ImageNet backbone to use as the luminance encoder
            ("convnext_tiny"), replacing the from-scratch encoder built from
            encoder_config_path. None keeps the historical behaviour.
        pretrained_encoder_stage_channels: per-stage widths the backbone's features are
            projected to before the decoder sees them; see PretrainedLuminanceEncoder.
        quantized_chroma_head: constructor kwargs for QuantizedChromaHead, replacing the
            smoothing network's final 2-channel projection with a distribution over ab cells.
            The smoothing config must then stop at the head's in_channels rather than at 2 --
            see smoothing_net_v4.json.
        cluster_color_prior: constructor kwargs for ClusterColorPrior, or None to leave the
            colour queries unconditioned. Requires a pretrained encoder, since the prior reads
            the same frozen backbone rather than running a second one.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.color_decoder_layers = color_decoder_layers

        self.luminance_encoder = self._buildLuminanceEncoder(
            pretrained_encoder, encoder_module, encoder_config_path,
            pretrained_encoder_stage_channels
        )
        encoder_channels = self.luminance_encoder.getDownsampleChannels()

        self.pixel_decoder = PixelDecoder(
            encoder_channels=encoder_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        self.pixel_decoder_layers = self.pixel_decoder.num_layers

        self.color_decoder = MultiScaleColorDecoder(
            color_dim=color_dim,
            embed_dim=embed_dim,
            output_dim=color_decoder_output_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_layers=color_decoder_layers,
            memory_size=memory_size
        )

        if color_decoder_output_dim != embed_dim:
            raise ValueError(
                f"color_decoder_output_dim ({color_decoder_output_dim}) must equal embed_dim "
                f"({embed_dim}): the colour decoder's output is contracted against the pixel "
                f"decoder's features, which are embed_dim wide"
            )

        self.smoothing_layers = self.loadSmoothingNetwork(smoothing_config_path)

        self.chroma_head = (
            QuantizedChromaHead(**quantized_chroma_head)
            if quantized_chroma_head is not None else None
        )

        self.cluster_color_prior = None
        if cluster_color_prior is not None:
            if pretrained_encoder is None:
                raise ValueError(
                    "cluster_color_prior needs a pretrained_encoder: it conditions on the same "
                    "frozen backbone's features rather than running a second backbone"
                )
            feature_dim = sum(self.luminance_encoder.backbone.getStageChannels())
            self.cluster_color_prior = ClusterColorPrior(
                feature_dim=feature_dim, output_dim=color_dim,
                backbone_variant=self.luminance_encoder.backbone._variant,
                **cluster_color_prior
            )

    @staticmethod
    def _buildLuminanceEncoder(pretrained_encoder: Optional[str],
                               encoder_module: Optional[nn.Module],
                               encoder_config_path: Optional[str],
                               stage_channels: Optional[List[int]]) -> nn.Module:
        """Pick the frozen pretrained encoder or the from-scratch one built from JSON.

        The variant is named in `pretrained_encoder` itself ("convnext_tiny", "convnext_small",
        "convnext_base") so a checkpoint's architecture file states which backbone produced it;
        a separate flag could drift from the weights it describes.
        """
        if pretrained_encoder is None:
            return LuminanceEncoder(encoder_module, encoder_config_path)
        prefix = "convnext_"
        if not pretrained_encoder.startswith(prefix):
            raise ValueError(
                f"unsupported pretrained_encoder {pretrained_encoder!r}, expected one of "
                f"{[prefix + name for name in sorted(CONVNEXT_VARIANTS)]}"
            )
        variant = pretrained_encoder[len(prefix):]
        if variant not in CONVNEXT_VARIANTS:
            raise ValueError(
                f"unsupported pretrained_encoder {pretrained_encoder!r}, expected one of "
                f"{[prefix + name for name in sorted(CONVNEXT_VARIANTS)]}"
            )
        return PretrainedLuminanceEncoder(stage_channels=stage_channels, variant=variant)

    def loadSmoothingNetwork(self, config_path: str) -> nn.Module:
        if config_path is None:
            default_config = [
                {"type": "Conv2d", "params": {"in_channels": self.memory_size, "out_channels": 64, "kernel_size": 3, "padding": 1}},
                {"type": "BatchNorm2d", "params": {"num_features": 64}},
                {"type": "GELU", "params": {}},
                {"type": "Conv2d", "params": {"in_channels": 64, "out_channels": 32, "kernel_size": 3, "padding": 1}},
                {"type": "BatchNorm2d", "params": {"num_features": 32}},
                {"type": "GELU", "params": {}},
                {"type": "Conv2d", "params": {"in_channels": 32, "out_channels": 1, "kernel_size": 3, "padding": 1}}
            ]
            return createLayersFromConfig(default_config, self)

        try:
            with open(config_path, 'r') as f:
                architecture = json.load(f)
            return createLayersFromConfig(architecture, self)
        except Exception as e:
            raise RuntimeError(f"Failed to load smoothing network from {config_path}: {e}")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        batch_size, channels, h, w = img.shape

        if channels == 3:
            luminance = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        elif channels == 1:
            luminance = img
        else:
            raise ValueError("Input must be 1-channel (luminance) or 3-channel (RGB)")

        encoder_outputs = self.luminance_encoder(luminance)

        color_prior = None
        if self.cluster_color_prior is not None:
            color_prior = self.cluster_color_prior(self.luminance_encoder.backbone(luminance))

        pixel_decoder_outputs, final_pixel_output = self.pixel_decoder(encoder_outputs)  # final_pixel_output: [batch_size, embed_dim, H, W]

        color_decoder_output = self.color_decoder(pixel_decoder_outputs, color_prior)  # [batch_size, memory_size, embed_dim]

        output = torch.einsum("bqc,bchw->bqhw", color_decoder_output, final_pixel_output)  # [batch_size, memory_size, H, W]
        output = self.smoothing_layers(output)  # [batch_size, 1, H, W]

        if self.chroma_head is not None:
            # The head returns ab like the regression path does, so every consumer downstream --
            # the loss's LPIPS term, the gate, visualisation -- is unchanged. The logits it kept
            # are what the classification term reads.
            output = self.chroma_head(output)

        return output


class VAE(nn.Module):
    def __init__(
        self,
        encoder: NeuralNetwork,
        decoder: NeuralNetwork,
        latent_dimensions: int = 128,
        input_channels: int = 3,
        input_image_size: int = 64
    ):
        super().__init__()
        self._latent_dimensions = latent_dimensions

        self._encoder = encoder
        self._decoder = decoder

        # Compute flattened size dynamically based on the last encoder layer
        with torch.no_grad():
            dummy_input = torch.zeros(2, input_channels, input_image_size, input_image_size)
            encoded_output: torch.Tensor = self._encoder(dummy_input)

            if len(encoded_output.shape) == 2:  # Already flattened: [batch, features]
                self._flattened_feature_size = encoded_output.size(1)
                self._encoder_output_shape = None
            else:  # Has spatial dimensions: [batch, channels, height, width]
                elements_per_sample = encoded_output.numel() // encoded_output.size(0)
                self._flattened_feature_size = elements_per_sample
                self._encoder_output_shape = encoded_output.shape[1:]  # Saving shape for decoder

        # Latent space projection layers
        self._mean_projection = nn.Linear(self._flattened_feature_size, latent_dimensions)
        self._log_variance_projection = nn.Linear(self._flattened_feature_size, latent_dimensions)
        self._latent_to_features = nn.Linear(latent_dimensions, self._flattened_feature_size)

    def forward(self, input_images: torch.Tensor):
        batch_size = input_images.size(0)

        encoded_features: torch.Tensor = self._encoder(input_images)

        if len(encoded_features.shape) > 2:
            flattened_features = encoded_features.view(batch_size, -1)
        else:
            flattened_features = encoded_features

        # Compute latent distribution parameters
        latent_mean = self._mean_projection(flattened_features)
        latent_log_variance = self._log_variance_projection(flattened_features)

        # Reparameterization trick
        latent_standard_deviation = torch.exp(0.5 * latent_log_variance)
        random_noise = torch.randn_like(latent_standard_deviation)
        latent_sample = latent_mean + random_noise * latent_standard_deviation

        decoded_features: torch.Tensor = self._latent_to_features(latent_sample)

        if self._encoder_output_shape is not None:
            reshaped_features = decoded_features.view(batch_size, *self._encoder_output_shape)
        else:
            reshaped_features = decoded_features

        reconstructed_images = self._decoder(reshaped_features)

        return reconstructed_images, latent_mean, latent_log_variance


class ConvAttenColorizationNetwork(nn.Module):
    # Ordered channel layouts: concatenation order defines the output channel order
    # (R, G, B for an RGB image, A, B for LAB chroma channels)
    _SUPPORTED_MODEL_SETS = (
        ("red_model", "green_model", "blue_model"),
        ("a_model", "b_model"),
    )

    def __init__(
        self,
        pretrained_models_config: Dict[str, Dict[str, str]],
        trainable_network: nn.Module,  # Generic trainable network
        pretrained_input_channels: int = 1,
        concatenate_input: Optional[bool] = None,
    ):
        super().__init__()

        self._color_model_names = self._resolve_model_names(pretrained_models_config.keys())

        self._pretrained_models_config = pretrained_models_config
        self._pretrained_input_channels = pretrained_input_channels
        self._concatenate_input: bool = self._resolveConcatenateInput(concatenate_input)

        self._pretrained_models = nn.ModuleDict()
        self._load_pretrained_models()

        self._trainable_network = trainable_network

    def train(self, mode: bool = True):
        # Frozen pretrained submodules must stay in eval mode so their
        # dropout/normalization layers behave identically in train and val phases
        super().train(mode)
        self._pretrained_models.eval()
        return self

    def _resolveConcatenateInput(self, concatenate_input: Optional[bool]) -> bool:
        """Decide whether the model input is stacked in front of the submodules' predictions.

        The chroma-only (a, b) layout has always needed it: without luminance the trainable
        network would see two channels that do not form an image. The RGB layout does not
        *need* it, but the trainable network is strictly better off with it -- the frozen
        extractors are the only source of structure otherwise, so any detail they dropped is
        unrecoverable. `None` keeps the historical per-layout behaviour so existing
        checkpoints load; pass an explicit bool to opt in or out.
        """
        if concatenate_input is not None:
            return concatenate_input
        return "a_model" in self._color_model_names

    def _resolve_model_names(self, configured_models) -> List[str]:
        for model_set in self._SUPPORTED_MODEL_SETS:
            if set(model_set).issubset(configured_models):
                return list(model_set)
        raise ValueError(
            f"Configured models {set(configured_models)} must include one of: "
            f"{[set(model_set) for model_set in self._SUPPORTED_MODEL_SETS]}"
        )

    def _load_pretrained_models(self):
        for model_name, model_config in self._pretrained_models_config.items():
            try:
                with open(model_config["architecture_path"], 'r') as f:
                    architecture = json.load(f)
                model = NeuralNetwork(architecture)
                if "weights_path" in model_config and model_config["weights_path"]:
                    state_dict = torch.load(model_config["weights_path"], map_location='cpu', weights_only=True)
                    model.load_state_dict(state_dict)
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
                self._pretrained_models[model_name] = model
            except Exception as e:
                raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def _separateAuxiliaryChannels(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split off conditioning channels stacked after the ones the pretrained models expect.

        Edge maps and SAM segmentation encodings both arrive this way, and both are meant for
        the trainable network alone: the R/G/B extractors are frozen, were trained on
        luminance only, and could not read an extra channel even if it were handed to them.
        Splitting here (rather than in the dataset) keeps the dataset's job to "produce one
        input tensor" and puts the knowledge of who consumes what in the module that routes it.
        """
        if x.size(1) <= self._pretrained_input_channels:
            return x, None
        return x[:, :self._pretrained_input_channels], x[:, self._pretrained_input_channels:]

    def _generate_color_channels(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]

        with torch.no_grad():
            color_channels = [self._pretrained_models[name](x) for name in self._color_model_names]

        color_channels = [
            nn.functional.interpolate(channel, size=(height, width), mode='bicubic')
            if channel.shape[-2:] != (height, width) else channel
            for channel in color_channels
        ]

        return torch.cat(color_channels, dim=1)  # [B, num_models, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_input, auxiliary = self._separateAuxiliaryChannels(x)
        initial_colors = self._generate_color_channels(model_input)
        if self._concatenate_input:
            # Stack the pristine input (luminance) in front of the predictions, so the
            # trainable network sees full LAB / luminance+RGB rather than the frozen
            # extractors' output alone
            initial_colors = torch.cat([model_input, initial_colors], dim=1)
        if auxiliary is not None:
            initial_colors = torch.cat([initial_colors, auxiliary], dim=1)
        final_output = self._trainable_network(initial_colors)
        return final_output


def createGroupNorm(norm_groups: Optional[int], channels: int) -> nn.Module:
    """Group normalisation for a UNet block, or a no-op when normalisation is disabled.

    `None` (the default everywhere) yields `nn.Identity`, so a block built without the
    parameter is numerically identical to the pre-normalisation version and keeps loading
    old checkpoints. `norm_groups` must divide `channels`.
    """
    if norm_groups is None:
        return nn.Identity()
    if channels % norm_groups != 0:
        raise ValueError(
            f"norm_groups ({norm_groups}) must divide the channel count ({channels})"
        )
    return nn.GroupNorm(norm_groups, channels)


def createDropout2d(dropout: Optional[float]) -> nn.Module:
    """Channel dropout for a UNet block, or a no-op when it is disabled.

    Belongs inside the block rather than as a standalone layer in the architecture JSON:
    `UNetWithSkipConnections` buckets layers by type and applies everything that is not an
    encoder/decoder/bottleneck *after* the whole decoder, so a `Dropout2d` entry in the JSON
    lands at the output regardless of where it was written.
    """
    if dropout is None or dropout == 0.0:
        return nn.Identity()
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    return nn.Dropout2d(dropout)


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True, block_name: str = "",
                 norm_groups: Optional[int] = None):
        super().__init__()
        self._block_name = block_name
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = createGroupNorm(norm_groups, out_channels)
        self.norm2 = createGroupNorm(norm_groups, out_channels)
        self.activation = nn.GELU()

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if downsample:
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        if residual.shape == x.shape:
            x = x + residual

        x = self.activation(x)
        self._skip_features = x

        if self.downsample and self.downsample_conv is not None:
            return self.downsample_conv(x)
        return x


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True,
                 skip_connection: str = "", block_name: str = "",
                 norm_groups: Optional[int] = None, dropout: Optional[float] = None,
                 semantic_channels: int = 0):
        """
        semantic_channels: width of the semantic map concatenated onto this block's input, or 0
            for none. 0 leaves the block numerically identical to the version without semantic
            injection, so existing checkpoints keep loading.
        """
        super().__init__()
        self._skip_connection_name = skip_connection
        self._block_name = block_name
        self.upsample = upsample
        self.semantic_channels = semantic_channels

        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample_conv = None

        # Applied after the skip connection is added, so the semantic map meets features that
        # already carry the encoder's detail rather than competing with it for the sum.
        self.semantic_merge = (
            nn.Conv2d(out_channels + semantic_channels, out_channels, kernel_size=1)
            if semantic_channels > 0 else None
        )
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = createGroupNorm(norm_groups, out_channels)
        self.norm2 = createGroupNorm(norm_groups, out_channels)
        self.dropout = createDropout2d(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, encoder_features: Dict[str, torch.Tensor],
                semantic_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.upsample and self.upsample_conv is not None:
            x = self.upsample_conv(x)
            x = self.activation(x)

        if self._skip_connection_name and self._skip_connection_name in encoder_features:
            skip_features = encoder_features[self._skip_connection_name]
            if skip_features.shape[-2:] != x.shape[-2:]:
                raise ValueError(f"Skip connection dimension mismatch: {skip_features.shape} vs {x.shape}")
            x = x + skip_features

        if self.semantic_channels > 0:
            if semantic_features is None:
                raise ValueError(
                    f"decoder block {self._block_name!r} was built with semantic_channels="
                    f"{self.semantic_channels} but received no semantic features"
                )
            if semantic_features.shape[-2:] != x.shape[-2:]:
                raise ValueError(
                    f"semantic feature map {tuple(semantic_features.shape[-2:])} does not match "
                    f"decoder block {self._block_name!r} at {tuple(x.shape[-2:])}; check that "
                    f"the encoder emits this block's stride"
                )
            x = self.semantic_merge(torch.cat([x, semantic_features], dim=1))

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        return self.dropout(x)


class UNetBottleneck(nn.Module):
    def __init__(self, channels: int, bottleneck_channels: int = 512,
                 norm_groups: Optional[int] = None, dropout: Optional[float] = None,
                 semantic_channels: int = 0):
        """
        semantic_channels: width of the semantic feature map concatenated onto the bottleneck
            input by UNetWithSkipConnections. 0 (the default) leaves the block numerically
            identical to the version without semantic injection, so old checkpoints load.
            The block's OUTPUT width stays `channels` either way, so the decoder is unaffected.
        """
        super().__init__()
        self.semantic_channels = semantic_channels
        input_channels = channels + semantic_channels

        self.conv1 = nn.Conv2d(input_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, kernel_size=3, padding=1)
        self.norm1 = createGroupNorm(norm_groups, bottleneck_channels)
        self.norm2 = createGroupNorm(norm_groups, bottleneck_channels)
        self.norm3 = createGroupNorm(norm_groups, channels)
        self.dropout = createDropout2d(dropout)
        self.activation = nn.GELU()

        self.residual1 = nn.Conv2d(input_channels, bottleneck_channels, kernel_size=1) if input_channels != bottleneck_channels else None
        self.residual2 = nn.Conv2d(bottleneck_channels, channels, kernel_size=1) if bottleneck_channels != channels else None

    def forward(self, x: torch.Tensor,
                semantic_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.semantic_channels > 0:
            if semantic_features is None:
                raise ValueError(
                    f"bottleneck was built with semantic_channels={self.semantic_channels} but "
                    f"received no semantic features"
                )
            if semantic_features.shape[-2:] != x.shape[-2:]:
                raise ValueError(
                    f"semantic feature map {tuple(semantic_features.shape[-2:])} does not match "
                    f"the bottleneck resolution {tuple(x.shape[-2:])}; check that the encoder's "
                    f"total downsampling equals the semantic encoder's output_stride"
                )
            x = torch.cat([x, semantic_features], dim=1)

        residual1 = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        if self.residual1 is not None:
            residual1 = self.residual1(residual1)
        if residual1.shape == x.shape:
            x = x + residual1

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        residual2 = x
        x = self.conv3(x)
        x = self.norm3(x)

        if self.residual2 is not None:
            residual2 = self.residual2(residual2)
        if residual2.shape == x.shape:
            x = x + residual2

        return self.dropout(self.activation(x))


class UNetWithSkipConnections(nn.Module):
    def __init__(self, layers_config: List[Dict[str, Any]],
                 semantic_encoder: Optional[Dict[str, Any]] = None,
                 semantic_input_channel: int = 0,
                 multi_scale_semantic_encoder: Optional[Dict[str, Any]] = None,
                 cluster_color_prior: Optional[Dict[str, Any]] = None,
                 quantized_chroma_head: Optional[Dict[str, Any]] = None):
        """
        semantic_encoder: constructor kwargs for PretrainedSemanticEncoder, or None to build a
            plain UNet. When given, its fused feature map is concatenated onto the bottleneck
            input, and the matching UNetBottleneck must declare the same width via
            semantic_channels.
        multi_scale_semantic_encoder: constructor kwargs for MultiScaleSemanticEncoder, the
            alternative that feeds every decoder level rather than the bottleneck alone. Each
            block declaring semantic_channels > 0 is handed the pyramid level matching its own
            resolution, so the encoder's output_strides must cover the strides the decoder
            actually runs at. Mutually exclusive with semantic_encoder.
        cluster_color_prior: constructor kwargs for ClusterColorPrior, applied to the bottleneck
            output through FiLM. Requires multi_scale_semantic_encoder, whose frozen backbone it
            shares rather than running a second one.
        semantic_input_channel: which input channel carries the pristine luminance the semantic
            encoder is to run on. Explicit rather than assumed, because nothing downstream can
            detect a wrong choice: the backbone will happily consume a predicted red channel
            and produce plausible-looking features. For rgb_merge the input is
            [luminance, R, G, B] and the answer is 0, but that holds only while
            CONCATENATE_INPUT is on -- with it off channel 0 is a colour prediction instead.
        """
        super().__init__()
        self.semantic_input_channel = semantic_input_channel
        self.encoder_blocks = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()
        self.other_layers = nn.ModuleList()
        self.bottleneck = None

        if semantic_encoder is not None and multi_scale_semantic_encoder is not None:
            raise ValueError(
                "semantic_encoder and multi_scale_semantic_encoder are mutually exclusive: "
                "the second is the multi-level replacement for the first"
            )
        self.semantic_encoder = (
            PretrainedSemanticEncoder(**semantic_encoder) if semantic_encoder is not None
            else None
        )
        self.multi_scale_semantic_encoder = (
            MultiScaleSemanticEncoder(**multi_scale_semantic_encoder)
            if multi_scale_semantic_encoder is not None else None
        )

        self.cluster_color_prior = None
        self.prior_film = None
        self._prior_kwargs = cluster_color_prior

        for layer_config in layers_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]

            if layer_type == "UNetEncoderBlock":
                block_name = params.get("block_name", f"enc_{len(self.encoder_blocks)}")
                self.encoder_blocks[block_name] = UNetEncoderBlock(**params)
            elif layer_type == "UNetDecoderBlock":
                block_name = params.get("block_name", f"dec_{len(self.decoder_blocks)}")
                self.decoder_blocks[block_name] = UNetDecoderBlock(**params)
            elif layer_type == "UNetBottleneck":
                self.bottleneck = UNetBottleneck(**params)
                if cluster_color_prior is not None:
                    if self.multi_scale_semantic_encoder is None:
                        raise ValueError(
                            "cluster_color_prior needs multi_scale_semantic_encoder: it reads "
                            "that encoder's frozen backbone instead of running a second one"
                        )
                    backbone = self.multi_scale_semantic_encoder.backbone
                    prior_dim = params.get("channels")
                    self.cluster_color_prior = ClusterColorPrior(
                        feature_dim=sum(backbone.getStageChannels()),
                        output_dim=prior_dim,
                        backbone_variant=backbone._variant,
                        **cluster_color_prior
                    )
                    self.prior_film = FilmModulation(condition_dim=prior_dim,
                                                     channels=prior_dim)
            else:
                self.other_layers.append(NeuralNetworkLayer(layer_config))

    @staticmethod
    def _selectPyramidLevel(pyramid: Dict[int, torch.Tensor],
                            target_size: Tuple[int, int], block_name: str) -> torch.Tensor:
        """The pyramid level whose spatial size matches this decoder block's output."""
        for feature in pyramid.values():
            if tuple(feature.shape[-2:]) == tuple(target_size):
                return feature
        available = sorted(tuple(f.shape[-2:]) for f in pyramid.values())
        raise ValueError(
            f"decoder block {block_name!r} needs a semantic map at {tuple(target_size)} but the "
            f"encoder emits {available}; add the matching stride to output_strides"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = {}

        # Run before the encoder consumes x: the semantic branch needs the pristine luminance
        # channel of the network input (see semantic_input_channel).
        semantic_features = None
        semantic_pyramid: Dict[int, torch.Tensor] = {}
        color_prior: Optional[torch.Tensor] = None
        if self.semantic_encoder is not None or self.multi_scale_semantic_encoder is not None:
            channel = self.semantic_input_channel
            if channel >= x.shape[1]:
                raise ValueError(
                    f"semantic_input_channel {channel} is out of range for an input with "
                    f"{x.shape[1]} channels"
                )
            luminance = x[:, channel:channel + 1]
            if self.semantic_encoder is not None:
                semantic_features = self.semantic_encoder(luminance)
            else:
                stages = self.multi_scale_semantic_encoder.computeStages(luminance)
                semantic_pyramid = self.multi_scale_semantic_encoder.fusePyramid(stages)
                if self.cluster_color_prior is not None:
                    color_prior = self.cluster_color_prior(stages)

        for name, encoder_block in self.encoder_blocks.items():
            x = encoder_block(x)
            encoder_features[name] = encoder_block._skip_features

        if self.bottleneck is not None:
            x = self.bottleneck(x, semantic_features)
            if self.prior_film is not None and color_prior is not None:
                x = self.prior_film(x, color_prior)

        for name, decoder_block in self.decoder_blocks.items():
            # Matched by resolution, not by position: a block that does not upsample shares its
            # predecessor's stride, so counting blocks off would misalign the pyramid silently.
            block_semantic = None
            if decoder_block.semantic_channels > 0:
                target_size = ((x.shape[-2] * 2, x.shape[-1] * 2) if decoder_block.upsample
                               else (x.shape[-2], x.shape[-1]))
                block_semantic = self._selectPyramidLevel(semantic_pyramid, target_size, name)
            x = decoder_block(x, encoder_features, block_semantic)

        for layer in self.other_layers:
            x = layer(x)

        return x


class SegFormer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: List[int] = [4, 2, 1, 1],
        embed_dims: List[int] = [32, 64, 160, 256],
        patch_kernel_size: List[int] = [7, 3, 3, 3],
        patch_stride: List[int] = [4, 2, 2, 2],
        patch_padding: List[int] = [3, 1, 1, 1],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        num_heads: List[int] = [1, 2, 5, 8],
        depths: List[int] = [2, 2, 2, 2],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()
        self.segformer_encoder = MixVisionTransformer3D(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )

        reversed_embed_dims = embed_dims[::-1]
        self.segformer_decoder = SegFormerDecoderHead3D(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )
        self.apply(self._initWeights)

    def _initWeights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv3d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        original_size = x.shape[2:]
        features = self.segformer_encoder(x)
        c1, c2, c3, c4 = features
        output = self.segformer_decoder(c1, c2, c3, c4, original_size)
        return output


class PatchEmbedding3D(nn.Module):
    def __init__(
        self,
        in_channel: int = 4,
        embed_dim: int = 768,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.stride = stride

    def forward(self, x):
        patches = self.patch_embeddings(x)
        B, C, D, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches, (D, H, W)


class DWConv3D(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x, spatial_dims):
        B, N, C = x.shape
        D, H, W = spatial_dims
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP3D(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv3D(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_dims):
        x = self.fc1(x)
        x = self.dwconv(x, spatial_dims)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SegFormerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.attention_head_dim = embed_dim // num_heads

        self.multi_head_attention = MultiHeadAttention(
            query_dim=embed_dim,
            value_dim=embed_dim,
            key_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            use_causal_mask=False
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, spatial_dims):
        B, N, C = x.shape
        D, H, W = spatial_dims

        q = x
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            k = v = x_
        else:
            k = q
            v = q

        out = self.multi_head_attention(query=q, key=k, value=v)
        return out


class SegFormerBlock3D(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        sr_ratio: int = 2,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SegFormerAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attn_dropout=attn_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP3D(
            in_feature=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=proj_dropout
        )

    def forward(self, x, spatial_dims):
        x = x + self.attention(self.norm1(x), spatial_dims)
        x = x + self.mlp(self.norm2(x), spatial_dims)
        return x


class MixVisionTransformer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: List[int] = [8, 4, 2, 1],
        embed_dims: List[int] = [64, 128, 320, 512],
        patch_kernel_size: List[int] = [7, 3, 3, 3],
        patch_stride: List[int] = [4, 2, 2, 2],
        patch_padding: List[int] = [3, 1, 1, 1],
        mlp_ratios: List[int] = [2, 2, 2, 2],
        num_heads: List[int] = [1, 2, 5, 8],
        depths: List[int] = [2, 2, 2, 2],
    ):
        super().__init__()

        self.embed_1 = PatchEmbedding3D(
            in_channel=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=patch_kernel_size[0],
            stride=patch_stride[0],
            padding=patch_padding[0],
        )
        self.embed_2 = PatchEmbedding3D(
            in_channel=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=patch_kernel_size[1],
            stride=patch_stride[1],
            padding=patch_padding[1],
        )
        self.embed_3 = PatchEmbedding3D(
            in_channel=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=patch_kernel_size[2],
            stride=patch_stride[2],
            padding=patch_padding[2],
        )
        self.embed_4 = PatchEmbedding3D(
            in_channel=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=patch_kernel_size[3],
            stride=patch_stride[3],
            padding=patch_padding[3],
        )

        self.tf_block1 = nn.ModuleList([
            SegFormerBlock3D(
                embed_dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                sr_ratio=sr_ratios[0]
            ) for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.tf_block2 = nn.ModuleList([
            SegFormerBlock3D(
                embed_dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                sr_ratio=sr_ratios[1]
            ) for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.tf_block3 = nn.ModuleList([
            SegFormerBlock3D(
                embed_dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                sr_ratio=sr_ratios[2]
            ) for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.tf_block4 = nn.ModuleList([
            SegFormerBlock3D(
                embed_dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                sr_ratio=sr_ratios[3]
            ) for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        outputs = []

        x, spatial_dims1 = self.embed_1(x)
        B, N, C = x.shape
        D1, H1, W1 = spatial_dims1
        for blk in self.tf_block1:
            x = blk(x, (D1, H1, W1))
        x = self.norm1(x)
        x = x.reshape(B, D1, H1, W1, -1).permute(0, 4, 1, 2, 3).contiguous()
        outputs.append(x)

        x, spatial_dims2 = self.embed_2(x)
        B, N, C = x.shape
        D2, H2, W2 = spatial_dims2
        for blk in self.tf_block2:
            x = blk(x, (D2, H2, W2))
        x = self.norm2(x)
        x = x.reshape(B, D2, H2, W2, -1).permute(0, 4, 1, 2, 3).contiguous()
        outputs.append(x)

        x, spatial_dims3 = self.embed_3(x)
        B, N, C = x.shape
        D3, H3, W3 = spatial_dims3
        for blk in self.tf_block3:
            x = blk(x, (D3, H3, W3))
        x = self.norm3(x)
        x = x.reshape(B, D3, H3, W3, -1).permute(0, 4, 1, 2, 3).contiguous()
        outputs.append(x)

        x, spatial_dims4 = self.embed_4(x)
        B, N, C = x.shape
        D4, H4, W4 = spatial_dims4
        for blk in self.tf_block4:
            x = blk(x, (D4, H4, W4))
        x = self.norm4(x)
        x = x.reshape(B, D4, H4, W4, -1).permute(0, 4, 1, 2, 3).contiguous()
        outputs.append(x)

        return outputs


class MLP3DDecoder(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = self.bn(x)
        return x


class SegFormerDecoderHead3D(nn.Module):
    def __init__(
        self,
        input_feature_dims: List[int] = [512, 320, 128, 64],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear_c4 = MLP3DDecoder(input_dim=input_feature_dims[0], embed_dim=decoder_head_embedding_dim)
        self.linear_c3 = MLP3DDecoder(input_dim=input_feature_dims[1], embed_dim=decoder_head_embedding_dim)
        self.linear_c2 = MLP3DDecoder(input_dim=input_feature_dims[2], embed_dim=decoder_head_embedding_dim)
        self.linear_c1 = MLP3DDecoder(input_dim=input_feature_dims[3], embed_dim=decoder_head_embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_pred = nn.Conv3d(decoder_head_embedding_dim, num_classes, kernel_size=1)

    def forward(self, c1, c2, c3, c4, input_size):
        n = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4]).contiguous()
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4]).contiguous()
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4]).contiguous()
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4]).contiguous()

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        x = F.interpolate(x, size=input_size, mode="trilinear", align_corners=False)

        return x
