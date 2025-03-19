import torch
import torch.nn as nn
from . import register_connector
from .base import Connector
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum
import math


class PerceiverResampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size #2560
        depth = config.num_resampler_layers #3
        num_latents = config.num_queries #512
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        self.linear = nn.Linear(config.vision_hidden_size, config.hidden_size)
        self.position_encoding = PositionalEncoding(dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=64, heads=8),
                        FeedForward(dim=dim, mult=4),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        b, v = x.shape[:2]
        x = self.linear(x) #torch.Size([bs, 728*16, 2560])
        
        position_encoding = self.position_encoding(v).to(device='cuda', dtype=x.dtype) # [1, seq_len, d_model]
        x = x + position_encoding

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=1) #torch.Size([bs, 1, 512, 2560])

        x = x.unsqueeze(1)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents).squeeze(1)

    
@register_connector('naiveresampler')    
class ResamplerConnectorWithPE(Connector):
    def __init__(self, config):
        super().__init__()

        self._connector = PerceiverResampler(config)


# =================================resampler related =================================
def exists(val):
    return val is not None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len):
        pe = torch.zeros(seq_len, self.d_model, device='cuda', dtype=torch.float16)
        position = torch.arange(0, seq_len, device='cuda').unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device='cuda').float() * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)
    
