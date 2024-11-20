import torch
import torch.nn as nn
from . import register_connector
from .base import Connector
import torch
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
        self.position_encoding = PositionalEncoding(dim, num_latents)

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

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=1) #torch.Size([bs, 1, 512, 2560])
        position_encoding = self.position_encoding(x) 
        #print("position_encoding:",position_encoding.shape) #torch.Size([1, 512, 2560])
        latents = latents + position_encoding

        x = x.unsqueeze(1)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents).squeeze(1)

    
@register_connector('resampler')    
class ResamplerConnector(Connector):
    def __init__(self, config):
        super().__init__()

        self._connector = PerceiverResampler(config)


# =================================resampler related =================================
def exists(val):
    return val is not None

"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, len=512):
        super().__init__()
        #self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(len, d_model)
        position = torch.arange(0, len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:, :seq_len]
        #x = x + self.pe[:, :x.size(1)]
        #return self.dropout(x)
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        self.pool = nn.AdaptiveAvgPool1d(max_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [bs, d_model, seq_len]
        x = self.pool(x)  # [bs, d_model, max_len]
        x = x.permute(0, 2, 1)  # [bs, max_len, d_model]
        return self.pe[:, :x.size(1), :]


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
    
