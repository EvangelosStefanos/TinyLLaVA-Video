import torch
from . import register_connector
from .base import Connector


class NullResamplerModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        return

    def forward(self, x):
        return x

    
@register_connector('nullresampler')    
class NullResampler(Connector):
    def __init__(self, config):
        super().__init__()
        self._connector = NullResamplerModel(config)
        return

