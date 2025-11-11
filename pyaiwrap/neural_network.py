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
    def __init__(self, patch_size=4, embed_dim=384, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        B, num_patches, embed_dim = x.shape
        h = w = int(num_patches ** 0.5)

        x = x.transpose(1, 2).view(B, embed_dim, h, w)
        x = self.proj(x)

        x = F.interpolate(
            x,
            scale_factor=self.patch_size,
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
        return x  # [B, out_channels, H, W]


class TransformerNet(nn.Module):
    """
    Complete Transformer Network: N Encoder Layers → N Decoder Layers → Output

    Architecture:
    - N Encoder layers (self-attention only, bidirectional)
    - N Decoder layers (causal self-attention + cross-attention to encoder)
    - Linear projection + Softmax

    Follows "Attention is All You Need" architecture:
    - Encoder processes input sequence
    - Decoder processes target sequence (autoregressive) with cross-attention to encoder
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 output_dim: Optional[int] = None,
                 only_use_encoder: bool = False,
                 use_decoder_masking: bool = True):
        """
        Args:
            dim: Dimension of features throughout the network
            num_heads: Number of attention heads
            mlp_ratio: Expansion ratio for MLP
            dropout: Dropout probability
            num_layers: Number of encoder AND decoder layers (N each)
            output_dim: Output dimension. If None, uses dim (for same shape output)
        """
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim is not None else dim
        self._only_use_encoder = only_use_encoder

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
                        use_causal_mask=use_decoder_masking  # Causal mask - only for autoregressive!
                    ),
                    'norm1': nn.LayerNorm(dim),

                    # Queries from decoder, Keys and Values from encoder output
                    'cross_attention': MultiHeadAttention(
                        query_dim=dim,
                        key_dim=dim,  # From encoder
                        value_dim=dim,  # From encoder
                        embed_dim=dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_causal_mask=False
                    ),
                    'norm2': nn.LayerNorm(dim),

                    # Position-wise Feed-Forward Network (MLP)
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

            # Learnable decoder start token
            self.decoder_start_token = nn.Parameter(torch.randn(1, 1, dim))

        self.output_projection = nn.Linear(dim, self.output_dim)
        self._output = nn.Identity()

    def forward(self,
                encoder_input: torch.Tensor,
                decoder_input: Optional[torch.Tensor] = None,
                seq_length: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through encoder-decoder transformer.

        Args:
            encoder_input: [B, N_enc, D] - input to encoder (with positional encoding)
            decoder_input: [B, N_dec, D] - optional decoder input (for teacher forcing)
            seq_length: int - if decoder_input not provided, length of sequence to generate

        Returns:
            output: [B, N_dec, output_dim] - output sequence
        """
        batch_size = encoder_input.shape[0]

        # encoder: N layers
        encoder_output = encoder_input
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)
        # encoder_output: [B, N_enc, D] - "memory" for cross-attention
        if not self._only_use_encoder:
            # decoder: prepare input
            if decoder_input is not None:
                # Teacher forcing: use provided decoder input
                decoder_output = decoder_input
            else:
                # start with token that should be learned and repeat for sequence length
                if seq_length is None:
                    seq_length = encoder_input.shape[1]  # Same length as encoder by default

                decoder_output = self.decoder_start_token.repeat(batch_size, seq_length, 1)

                pos_encoding = distancePositionalEncoding(seq_length, 1, self.dim, encoder_input.device)
                decoder_output = decoder_output + pos_encoding

            # ===== DECODER: N layers =====
            for decoder_layer in self.decoder_layers:
                # 1. Self-attention (with optional causal mask for autoregressive)
                self_attn_out = decoder_layer['self_attention'](query=decoder_output)
                decoder_output = decoder_output + self_attn_out
                decoder_output = decoder_layer['norm1'](decoder_output)

                # 2. Cross-attention to encoder output
                cross_attn_out = decoder_layer['cross_attention'](
                    query=decoder_output,      # Queries from decoder
                    key=encoder_output,        # Keys from encoder memory
                    value=encoder_output       # Values from encoder memory
                )
                decoder_output = decoder_output + cross_attn_out
                decoder_output = decoder_layer['norm2'](decoder_output)

                # 3. Position-wise Feed-Forward (MLP)
                mlp_out = decoder_layer['mlp'](decoder_output)
                decoder_output = decoder_output + mlp_out
                decoder_output = decoder_layer['norm3'](decoder_output)

        # ===== OUTPUT PROJECTION =====
        if self._only_use_encoder:
            decoder_output = encoder_output
        output = self.output_projection(decoder_output)  # [B, N_dec or N_enc, output_dim]
        output = self._output(output)  # [B, N_dec or N_enc, output_dim]

        return output


class ColorizationTransformerNet(nn.Module):
    """
    Specialized transformer for grayscale to RGB/single-channel colorization.
    Uses encoder-decoder architecture with proper separation of grayscale context and color generation.
    Implements scheduled sampling to bridge training-inference gap.
    """

    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 patch_size: int = 16,
                 use_decoder_masking: bool = False,
                 only_use_encoder: bool = True,
                 output_channels: int = 3,
                 **kwargs):
        """
        Args:
            embed_dim: Dimension of patch embeddings
            num_heads: Number of attention heads
            mlp_ratio: Expansion ratio for MLP
            dropout: Dropout probability
            num_layers: Number of encoder AND decoder layers
            patch_size: Size of image patches
            use_decoder_masking: Whether to use causal masking in decoder
            output_channels: Number of output channels (1 for single channel, 3 for RGB)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.output_channels = output_channels
        self._only_use_encoder = only_use_encoder

        self.grayscale_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=1,  # Grayscale input
            embed_dim=embed_dim
        )

        self.color_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=output_channels,  # Dynamic output channels
            embed_dim=embed_dim
        )

        self.patch_upsample = PatchUpsample(
            patch_size=patch_size,
            embed_dim=output_channels,  # Dynamic output
            out_channels=output_channels
        )

        self.transformer = TransformerNet(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_layers=num_layers,
            output_dim=output_channels,
            use_decoder_masking=use_decoder_masking,
            only_use_encoder=only_use_encoder
        )

    def forward(self,
                grayscale_img: torch.Tensor,
                rgb_img: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for colorization with scheduled sampling.

        Args:
            grayscale_img: [B, 1, H, W] - input grayscale image
            target_rgb: [B, C, H, W] - target color image (for teacher forcing during training)

        Returns:
            output: [B, C, H, W] - colorized output image
        """
        batch_size, _, h, w = grayscale_img.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Image size ({h}x{w}) must be divisible by patch_size ({self.patch_size})")

        patch_h = h // self.patch_size
        patch_w = w // self.patch_size
        num_patches = patch_h * patch_w

        grayscale_patches = self.grayscale_embed(grayscale_img)  # [B, num_patches, embed_dim]
        grayscale_pos_encoding = distancePositionalEncoding(patch_h, patch_w, self.embed_dim, grayscale_img.device)
        encoder_input = grayscale_patches + grayscale_pos_encoding

        if self._only_use_encoder:
            decoder_input = None
            current_seq_len = None
        else:
            if rgb_img is None:
                initial_colors = self._get_initial_colors(grayscale_img)
                pred_patches = self.color_embed(initial_colors)
                decoder_input = pred_patches
            else:
                pred_patches = self.color_embed(rgb_img)
                decoder_input = pred_patches
            current_seq_len = decoder_input.size(1)
            color_pos_encoding = distancePositionalEncoding(patch_h, patch_w, self.embed_dim, grayscale_img.device)
            decoder_input = decoder_input + color_pos_encoding[:, :current_seq_len, :]

        color_embeddings = self.transformer(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            seq_length=current_seq_len
        )  # [B, seq_len, output_channels]

        output = self.patch_upsample(color_embeddings)  # [B, output_channels, H, W]

        return output

    def _get_initial_colors(self, grayscale_img: torch.Tensor) -> torch.Tensor:
        """
        Get initial color estimate with small variations to prevent mode collapse.
        Args:
            grayscale_img: [B, 1, H, W] input grayscale
        Returns:
            initial_colors: [B, C, H, W] initial color estimate
        """
        batch_size, _, h, w = grayscale_img.shape
        if self.output_channels == 1:
            noise = torch.randn_like(grayscale_img) * 0.05
            initial_colors = torch.clamp(grayscale_img + noise, 0, 1)
        else:
            base = grayscale_img
            noise_r = torch.randn_like(base) * 0.03
            noise_g = torch.randn_like(base) * 0.03
            noise_b = torch.randn_like(base) * 0.03

            initial_colors = torch.cat([
                torch.clamp(base + noise_r, 0, 1),      # Red channel
                torch.clamp(base + noise_g, 0, 1),      # Green channel
                torch.clamp(base + noise_b, 0, 1)       # Blue channel
            ], dim=1)

        return initial_colors


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
    """UNet encoder block with optional downsampling"""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True, block_name: str = ""):
        super().__init__()
        self._block_name = block_name

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.GELU())

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.GELU())

        self._conv_block = nn.Sequential(*layers)

        if downsample:
            self._downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.GELU()
            )
        else:
            self._downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._conv_block(x)

        self._skip_features = features

        if self._downsample is not None:
            return self._downsample(features)
        return features


class UNetDecoderBlock(nn.Module):
    """UNet decoder block with summed skip connections"""

    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True,
                 skip_connection: str = "", block_name: str = ""):
        super().__init__()
        self._skip_connection_name = skip_connection
        self._block_name = block_name

        if upsample:
            self._upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.GELU()
            )
            conv_input_channels = out_channels
        else:
            self._upsample = None
            conv_input_channels = in_channels

        self._conv_block = nn.Sequential(
            nn.Conv2d(conv_input_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, encoder_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._upsample is not None:
            x = self._upsample(x)

        if self._skip_connection_name and self._skip_connection_name in encoder_features:
            skip_features = encoder_features[self._skip_connection_name]

            if skip_features.shape[-2:] != x.shape[-2:]:
                skip_features = F.interpolate(
                    skip_features, size=x.shape[-2:], mode='bicubic', align_corners=False
                )
            x = x + skip_features

        return self._conv_block(x)


class UNetBottleneck(nn.Module):
    """Bottleneck layer at the bottom of UNet"""

    def __init__(self, channels: int, bottleneck_channels: int = 512):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(bottleneck_channels),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(bottleneck_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(x)


class UNetWithSkipConnections(nn.Module):
    """Complete UNet with summed skip connections"""

    def __init__(self, layers_config: List[Dict[str, Any]]):
        super().__init__()
        self._layers = nn.ModuleList()
        self._encoder_blocks = {}
        self._decoder_blocks = {}

        for layer_config in layers_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]

            if layer_type == "UNetEncoderBlock":
                block_name = params.get("block_name", f"enc_{len(self._encoder_blocks)}")
                layer = UNetEncoderBlock(**params)
                self._encoder_blocks[block_name] = layer
                self._layers.append(layer)
            elif layer_type == "UNetDecoderBlock":
                block_name = params.get("block_name", f"dec_{len(self._decoder_blocks)}")
                layer = UNetDecoderBlock(**params)
                self._decoder_blocks[block_name] = layer
                self._layers.append(layer)
            else:
                self._layers.append(NeuralNetworkLayer(layer_config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = {}
        for layer in self._layers:
            if isinstance(layer, UNetEncoderBlock):
                x = layer(x)
                encoder_features[layer._block_name] = layer._skip_features
            elif isinstance(layer, UNetDecoderBlock):
                x = layer(x, encoder_features)
            else:
                x = layer(x)
        return x
