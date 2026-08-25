"""
PromptIR: Prompting for All-in-One Blind Image Restoration.

Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
https://arxiv.org/abs/2306.13090

Adapted for HW4 — parameterized by config.ModelConfig.
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------- #
#  Layer Norm
# --------------------------------------------------------------------------- #

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# --------------------------------------------------------------------------- #
#  Gated-Dconv Feed-Forward Network (GDFN)
# --------------------------------------------------------------------------- #

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias,
                 large_kernel=False, large_kernel_size=7, simple_gate=False):
        super().__init__()
        self.simple_gate = simple_gate
        hidden_features = int(dim * ffn_expansion_factor)

        k = large_kernel_size if large_kernel else 3
        p = k // 2

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias,
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=k, stride=1, padding=p,
            groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias,
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        if self.simple_gate:
            # SimpleGate (NAFNet): just element-wise multiply, no activation
            x = x1 * x2
        else:
            x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# --------------------------------------------------------------------------- #
#  Multi-DConv Head Transposed Self-Attention (MDTA)
# --------------------------------------------------------------------------- #

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,
                 large_kernel=False, large_kernel_size=7):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        k = large_kernel_size if large_kernel else 3
        p = k // 2

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3,
            kernel_size=k, stride=1, padding=p,
            groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w",
            head=self.num_heads, h=h, w=w,
        )
        out = self.project_out(out)
        return out


# --------------------------------------------------------------------------- #
#  Transformer Block
# --------------------------------------------------------------------------- #

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 large_kernel=False, large_kernel_size=7, simple_gate=False):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,
                              large_kernel=large_kernel,
                              large_kernel_size=large_kernel_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias,
                               large_kernel=large_kernel,
                               large_kernel_size=large_kernel_size,
                               simple_gate=simple_gate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
#  Resizing modules
# --------------------------------------------------------------------------- #

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# --------------------------------------------------------------------------- #
#  Overlapped image patch embedding with 3x3 Conv
# --------------------------------------------------------------------------- #

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# --------------------------------------------------------------------------- #
#  Prompt Generation Block
# --------------------------------------------------------------------------- #

class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96,
                 lin_dim=192, spatial_prompt=False):
        super().__init__()
        self.spatial_prompt = spatial_prompt
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size),
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False,
        )

        if spatial_prompt:
            # Lightweight spatial gate with residual form:
            # output = prompt * (1 + gate), so clean regions (gate≈0) still get
            # the base prompt, while degraded regions (gate>0) get amplified.
            # This ensures clean areas ≈ identity when combined with the main path.
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(lin_dim, lin_dim // 4, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(lin_dim // 4, 1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = (
            prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        )
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear", align_corners=False)
        prompt = self.conv3x3(prompt)

        if self.spatial_prompt:
            # Residual spatial gate: prompt * (1 + gate)
            # gate ∈ [0,1] → multiplier ∈ [1,2]
            # Clean regions (gate≈0): prompt × 1 (base prompt preserved)
            # Degraded regions (gate≈1): prompt × 2 (amplified)
            gate = self.spatial_gate(x)  # (B, 1, H, W)
            prompt = prompt * (1.0 + gate)

        return prompt


# --------------------------------------------------------------------------- #
#  C3: Skip Attention — channel attention on encoder skip connections
# --------------------------------------------------------------------------- #

class SkipChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention for skip connections.

    Learns to re-weight channels so that clean-region detail is preserved
    while noisy/degraded channels are suppressed before being concatenated
    with the decoder path.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.body(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


# --------------------------------------------------------------------------- #
#  PromptIR — main model
# --------------------------------------------------------------------------- #

class PromptIR(nn.Module):
    """All-in-one image restoration with prompt learning.

    Args:
        cfg: A ``ModelConfig`` dataclass **or** keyword arguments matching the
             original constructor signature (for backward compatibility).
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list = None,
        num_refinement_blocks: int = 4,
        heads: list = None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        decoder: bool = False,
        # --- Optimization flags --- #
        spatial_prompt: bool = False,
        large_kernel: bool = False,
        large_kernel_size: int = 7,
        simple_gate: bool = False,
        decoder_dropout: float = 0.0,
        skip_attention: bool = False,
    ):
        super().__init__()

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        # Store extra kwargs for TransformerBlock
        tb_kwargs = dict(
            large_kernel=large_kernel,
            large_kernel_size=large_kernel_size,
            simple_gate=simple_gate,
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        self.decoder_dropout_rate = decoder_dropout

        # --- Prompt generators (decoder side) ----------------------------- #
        # Prompt bank dims are fixed architectural constants; lin_dim must
        # match the feature channels at each decoder level.
        _pd1, _pd2, _pd3 = 64, 128, 320  # prompt bank sizes
        self._pd1, self._pd2, self._pd3 = _pd1, _pd2, _pd3

        if self.decoder:
            self.prompt1 = PromptGenBlock(
                prompt_dim=_pd1, prompt_len=5, prompt_size=64,
                lin_dim=int(dim * 2 ** 1),  # decoder lvl2 channels
                spatial_prompt=spatial_prompt,
            )
            self.prompt2 = PromptGenBlock(
                prompt_dim=_pd2, prompt_len=5, prompt_size=32,
                lin_dim=int(dim * 2 ** 2),  # decoder lvl3 channels
                spatial_prompt=spatial_prompt,
            )
            self.prompt3 = PromptGenBlock(
                prompt_dim=_pd3, prompt_len=5, prompt_size=16,
                lin_dim=int(dim * 2 ** 3),  # latent channels
                spatial_prompt=spatial_prompt,
            )

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        # C3: Skip attention on encoder outputs before they enter decoder
        self.skip_attention = skip_attention
        if skip_attention:
            self.skip_attn1 = SkipChannelAttention(dim)                 # enc level1
            self.skip_attn2 = SkipChannelAttention(int(dim * 2 ** 1))   # enc level2
            self.skip_attn3 = SkipChannelAttention(int(dim * 2 ** 2))   # enc level3

        # C2: Decoder dropout
        self.dec_dropout = nn.Dropout2d(p=decoder_dropout) if decoder_dropout > 0 else None

        # --- Encoder ------------------------------------------------------ #
        self.reduce_noise_channel_1 = nn.Conv2d(
            dim + 64, dim, kernel_size=1, bias=bias,
        )
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim,
                    heads[0],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    **tb_kwargs,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)

        self.reduce_noise_channel_2 = nn.Conv2d(
            int(dim * 2 ** 1) + 128, int(dim * 2 ** 1), kernel_size=1, bias=bias,
        )
        d2 = int(dim * 2 ** 1)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    d2,
                    heads[1],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    **tb_kwargs,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.reduce_noise_channel_3 = nn.Conv2d(
            int(dim * 2 ** 2) + 256, int(dim * 2 ** 2), kernel_size=1, bias=bias,
        )
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    int(dim * 2 ** 2),
                    heads[2],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    **tb_kwargs,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2 ** 2))

        # --- Latent ------------------------------------------------------- #
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    int(dim * 2 ** 3),
                    heads[3],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    **tb_kwargs,
                )
                for _ in range(num_blocks[3])
            ]
        )

        # --- Decoder ------------------------------------------------------ #
        self.up4_3 = Upsample(int(dim * 2 ** 2))
        # up4_3 output (dim*2) + skip from encoder_level3 (dim*4)
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2) + int(dim * 4), int(dim * 2 ** 2), kernel_size=1, bias=bias,
        )
        # After concat with prompt3: latent (dim*8) + prompt (_pd3)
        self.noise_level3 = TransformerBlock(
            int(dim * 2 ** 3) + _pd3,
            heads[2],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
            **tb_kwargs,
        )
        self.reduce_noise_level3 = nn.Conv2d(
            int(dim * 2 ** 3) + _pd3, int(dim * 2 ** 2), kernel_size=1, bias=bias,
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    int(dim * 2 ** 2),
                    heads[2],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    **tb_kwargs,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias,
        )
        # After concat with prompt2: decoder_level3 out (dim*4) + prompt (_pd2)
        self.noise_level2 = TransformerBlock(
            int(dim * 2 ** 2) + _pd2,
            heads[2],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
            **tb_kwargs,
        )
        self.reduce_noise_level2 = nn.Conv2d(
            int(dim * 2 ** 2) + _pd2, int(dim * 2 ** 2), kernel_size=1, bias=bias,
        )

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    int(dim * 2 ** 1),
                    heads[1],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    **tb_kwargs,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        # After concat with prompt1: decoder_level2 out (dim*2) + prompt (_pd1)
        self.noise_level1 = TransformerBlock(
            int(dim * 2 ** 1) + _pd1,
            heads[2],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
            **tb_kwargs,
        )
        self.reduce_noise_level1 = nn.Conv2d(
            int(dim * 2 ** 1) + _pd1, int(dim * 2 ** 1), kernel_size=1, bias=bias,
        )

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    int(dim * 2 ** 1),
                    heads[0],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    **tb_kwargs,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    int(dim * 2 ** 1),
                    heads[0],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    **tb_kwargs,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias,
        )

    @classmethod
    def from_config(cls, cfg):
        """Construct from a ``ModelConfig`` dataclass."""
        return cls(
            inp_channels=cfg.inp_channels,
            out_channels=cfg.out_channels,
            dim=cfg.dim,
            num_blocks=cfg.num_blocks,
            num_refinement_blocks=cfg.num_refinement_blocks,
            heads=cfg.heads,
            ffn_expansion_factor=cfg.ffn_expansion_factor,
            bias=cfg.bias,
            LayerNorm_type=cfg.LayerNorm_type,
            decoder=cfg.decoder,
            # Optimization flags
            spatial_prompt=getattr(cfg, 'spatial_prompt', False),
            large_kernel=getattr(cfg, 'large_kernel', False),
            large_kernel_size=getattr(cfg, 'large_kernel_size', 7),
            simple_gate=getattr(cfg, 'simple_gate', False),
            decoder_dropout=getattr(cfg, 'decoder_dropout', 0.0),
            skip_attention=getattr(cfg, 'skip_attention', False),
        )

    def _dec_drop(self, x):
        """Apply decoder dropout if configured."""
        if self.dec_dropout is not None:
            return self.dec_dropout(x)
        return x

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # C3: Apply skip attention to encoder outputs
        if self.skip_attention:
            out_enc_level1 = self.skip_attn1(out_enc_level1)
            out_enc_level2 = self.skip_attn2(out_enc_level2)
            out_enc_level3 = self.skip_attn3(out_enc_level3)

        # --- Decoder with prompt injection -------------------------------- #
        if self.decoder:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        out_dec_level3 = self._dec_drop(out_dec_level3)  # C2: decoder dropout

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self._dec_drop(out_dec_level2)  # C2: decoder dropout

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        # Residual learning
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
