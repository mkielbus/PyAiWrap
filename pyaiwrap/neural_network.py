from typing import Any, Dict, List, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import json
from .utils import distancePositionalEncoding


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
        self.query_projection = nn.Linear(self.query_dim, self.embed_dim)
        self.key_projection = nn.Linear(self.key_dim, self.embed_dim)
        self.value_projection = nn.Linear(self.value_dim, self.embed_dim)

        # Output projection (project back to query dimension)
        self.output_projection = nn.Linear(self.embed_dim, query_dim)

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
                 only_use_encoder: bool = False,
                 output_channels: int = 3):
        super().__init__()

        self.embed_dim = embed_dim
        self.image_patches = num_image_patches
        self.output_channels = output_channels
        self.only_use_encoder = only_use_encoder
        self.num_color_tokens = num_color_tokens
        self.image_size = image_size

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
            use_layerwise_connections=True
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
        grayscale_pos_encoding = distancePositionalEncoding(patch_h, patch_w, self.embed_dim, grayscale_img.device)
        encoder_input = grayscale_patches + grayscale_pos_encoding

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
            color_pos = distancePositionalEncoding(color_patch_h, color_patch_w, self.embed_dim, grayscale_img.device)  # [num_color_tokens, embed_dim]

            decoder_input = color_embeddings.unsqueeze(0).repeat(batch_size, 1, 1) + color_pos  # [B, num_color_tokens, embed_dim]

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

            pos_encoding = distancePositionalEncoding(h_res, w_res, current_features.shape[-1], device)  # [1, h_res*w_res, decoder_channels[i]]
            transformer_input_with_pos = transformer_input + pos_encoding

            transformer_output = self.transformer_blocks[i](transformer_input_with_pos)  # [batch_size, h_res*w_res, decoder_channels[i]]

            b, n_patches, c = transformer_output.shape
            spatial_features = transformer_output.transpose(1, 2).view(b, c, h_res, w_res)  # [batch_size, decoder_channels[i], h_res, w_res]

            upsampled_features = self.upsample_layers[i](spatial_features)  # [batch_size, out_channels, h_up, w_up]

            b, c_up, h_up, w_up = upsampled_features.shape

            output_features = upsampled_features.view(b, c_up, -1).transpose(1, 2)  # [batch_size, h_up*w_up, c_up]

            feature_embed = self.feature_embeddings[i].weight.unsqueeze(0)  # [1, 1, out_channels]
            feature_embed = feature_embed.expand(batch_size, output_features.size(1), -1)  # [batch_size, h_up*w_up, out_channels]

            output_features_with_embed = output_features + feature_embed  # [batch_size, h_up*w_up, c_up]

            output_features_projected = self.output_projections[i](output_features_with_embed)  # [batch_size, h_up*w_up, embed_dim]

            pos_encoding = distancePositionalEncoding(h_up, w_up, self.embed_dim, device)  # [1, h_up*w_up, embed_dim]
            output_features_projected_with_pos = output_features_projected + pos_encoding  # [batch_size, h_up*w_up, embed_dim]
            pixel_decoder_outputs.append(output_features_projected_with_pos)

            current_features = output_features_with_embed  # [batch_size, h_up*w_up, c_up]

        b, n_patches, c_final = current_features.shape
        h_final = h_enc * (2 ** self.num_layers)
        w_final = w_enc * (2 ** self.num_layers)
        final_pixel_spatial = current_features.transpose(1, 2).view(b, c_final, h_final, w_final)  # [batch_size, embed_dim, h_final, w_final]

        return pixel_decoder_outputs, final_pixel_spatial


class MultiScaleColorDecoder(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 memory_size: int = 256):
        super().__init__()

        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.num_layers = num_layers

        self.color_embeddings = nn.Embedding(memory_size, embed_dim)
        nn.init.zeros_(self.color_embeddings.weight)

        self.memory_embeddings = nn.Embedding(memory_size, embed_dim)
        nn.init.zeros_(self.memory_embeddings.weight)

        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            decoder_block = nn.ModuleDict({
                'cross_attention': MultiHeadAttention(
                    query_dim=embed_dim,
                    key_dim=embed_dim,
                    value_dim=embed_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_causal_mask=False
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'self_attention': MultiHeadAttention(
                    query_dim=embed_dim,
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

    def forward(self, pixel_decoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        batch_size = pixel_decoder_outputs[0].shape[0]

        color_queries = self.color_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, memory_size, embed_dim]

        memory = self.memory_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, memory_size, embed_dim]

        color_decoder_output = color_queries

        for i, block in enumerate(self.decoder_blocks):
            pixel_features_idx = i % len(pixel_decoder_outputs)
            pixel_features = pixel_decoder_outputs[pixel_features_idx]  # [batch_size, num_patches, embed_dim]

            queries_with_pos = color_decoder_output + memory  # [batch_size, memory_size, embed_dim]

            cross_attn_out = block['cross_attention'](
                query=queries_with_pos,
                key=pixel_features,
                value=pixel_features
            )  # [batch_size, memory_size, embed_dim]

            color_decoder_output = color_decoder_output + cross_attn_out
            color_decoder_output = block['norm1'](color_decoder_output)

            self_attn_out = block['self_attention'](query=color_decoder_output + memory)  # [batch_size, memory_size, embed_dim]
            color_decoder_output = color_decoder_output + self_attn_out
            color_decoder_output = block['norm2'](color_decoder_output)

            mlp_out = block['mlp'](color_decoder_output)  # [batch_size, memory_size, embed_dim]
            color_decoder_output = color_decoder_output + mlp_out
            color_decoder_output = block['norm3'](color_decoder_output)

        return color_decoder_output  # [batch_size, memory_size, embed_dim]


class ColorMemoryTransformer(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 color_decoder_layers: int = 6,
                 memory_size: int = 256,
                 smoothing_config_path: str = None,
                 encoder_module: Optional[nn.Module] = None,
                 encoder_config_path: str = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.color_decoder_layers = color_decoder_layers

        self.luminance_encoder = LuminanceEncoder(encoder_module, encoder_config_path)
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
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_layers=color_decoder_layers,
            memory_size=memory_size
        )

        self.smoothing_layers = self.loadSmoothingNetwork(smoothing_config_path)

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

        pixel_decoder_outputs, final_pixel_output = self.pixel_decoder(encoder_outputs)  # final_pixel_output: [batch_size, embed_dim, H, W]

        color_decoder_output = self.color_decoder(pixel_decoder_outputs)  # [batch_size, memory_size, embed_dim]

        output = torch.einsum("bqc,bchw->bqhw", color_decoder_output, final_pixel_output)  # [batch_size, memory_size, H, W]
        output = self.smoothing_layers(output)  # [batch_size, 1, H, W]

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
    def __init__(
        self,
        pretrained_models_config: Dict[str, Dict[str, str]],
        trainable_network: nn.Module,  # Generic trainable network
    ):
        super().__init__()

        required_models = {"red_model", "green_model", "blue_model"}
        if not required_models.issubset(pretrained_models_config.keys()):
            raise ValueError(f"Required models: {required_models}")

        self._pretrained_models_config = pretrained_models_config

        self._pretrained_models = nn.ModuleDict()
        self._load_pretrained_models()

        self._trainable_network = trainable_network

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

    def _generate_color_channels(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        with torch.no_grad():
            red_channel = self._pretrained_models["red_model"](x)    # [B, 1, H, W]
            green_channel = self._pretrained_models["green_model"](x)  # [B, 1, H, W]
            blue_channel = self._pretrained_models["blue_model"](x)   # [B, 1, H, W]

        if red_channel.shape[-2:] != (height, width):
            red_channel = nn.functional.interpolate(red_channel, size=(height, width), mode='bicubic')
        if green_channel.shape[-2:] != (height, width):
            green_channel = nn.functional.interpolate(green_channel, size=(height, width), mode='bicubic')
        if blue_channel.shape[-2:] != (height, width):
            blue_channel = nn.functional.interpolate(blue_channel, size=(height, width), mode='bicubic')

        rgb_image = torch.cat([red_channel, green_channel, blue_channel], dim=1)  # [B, 3, H, W]
        return rgb_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_rgb = self._generate_color_channels(x)
        final_output = self._trainable_network(initial_rgb)
        return final_output


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True, block_name: str = ""):
        super().__init__()
        self._block_name = block_name
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
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
        x = self.activation(x)
        x = self.conv2(x)

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
                 skip_connection: str = "", block_name: str = ""):
        super().__init__()
        self._skip_connection_name = skip_connection
        self._block_name = block_name
        self.upsample = upsample

        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample_conv = None

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, encoder_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.upsample and self.upsample_conv is not None:
            x = self.upsample_conv(x)
            x = self.activation(x)

        if self._skip_connection_name and self._skip_connection_name in encoder_features:
            skip_features = encoder_features[self._skip_connection_name]
            if skip_features.shape[-2:] != x.shape[-2:]:
                raise ValueError(f"Skip connection dimension mismatch: {skip_features.shape} vs {x.shape}")
            x = x + skip_features

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)

        return x


class UNetBottleneck(nn.Module):
    def __init__(self, channels: int, bottleneck_channels: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

        self.residual1 = nn.Conv2d(channels, bottleneck_channels, kernel_size=1) if channels != bottleneck_channels else None
        self.residual2 = nn.Conv2d(bottleneck_channels, channels, kernel_size=1) if bottleneck_channels != channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual1 = x
        x = self.conv1(x)
        x = self.activation(x)

        if self.residual1 is not None:
            residual1 = self.residual1(residual1)
        if residual1.shape == x.shape:
            x = x + residual1

        x = self.conv2(x)
        x = self.activation(x)

        residual2 = x
        x = self.conv3(x)

        if self.residual2 is not None:
            residual2 = self.residual2(residual2)
        if residual2.shape == x.shape:
            x = x + residual2

        return self.activation(x)


class UNetWithSkipConnections(nn.Module):
    def __init__(self, layers_config: List[Dict[str, Any]]):
        super().__init__()
        self.encoder_blocks = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()
        self.other_layers = nn.ModuleList()
        self.bottleneck = None

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
            else:
                self.other_layers.append(NeuralNetworkLayer(layer_config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = {}

        for name, encoder_block in self.encoder_blocks.items():
            x = encoder_block(x)
            encoder_features[name] = encoder_block._skip_features

        if self.bottleneck is not None:
            x = self.bottleneck(x)

        for name, decoder_block in self.decoder_blocks.items():
            x = decoder_block(x, encoder_features)

        for layer in self.other_layers:
            x = layer(x)

        return x


class DynamicSpatialWeights(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.weight_predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)  # [B, hidden_channels, H, W]
        )

        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

        self.output_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.1)

    def forward(self, x):
        identity = x

        hidden = self.conv_in(x)  # [B, hidden_channels, H, W]

        spatial_weights = self.weight_predictor(x)  # [B, hidden_channels, H, W]

        weighted_hidden = hidden * spatial_weights

        correction = self.conv_out(weighted_hidden)

        return torch.clamp(identity + correction * self.output_scale, 0, 1)


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
