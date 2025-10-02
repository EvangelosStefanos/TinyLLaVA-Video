from . import register_vision_tower
from .base import VisionTower
from tinyllava.vjepa.model import VJEPAModel
from transformers import SiglipImageProcessor



@register_vision_tower('vjepa')
class VJEPAVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = VJEPAModel(cfg.vjepa)
        self._image_processor = SiglipImageProcessor.from_pretrained(cfg.model_name_or_path)
        self.config = cfg
        return


    def forward(self, x, **kwargs):
        # x: frames of shape [B, T, C, H, W]
        # return: tensor of shape [B, N, D] a compressed representation of tokens
        return self._vision_tower(x, **kwargs)
    
    
    def get_loss(self):
        return self._vision_tower.get_loss()
    
    
    def ema_update(self, momentum_scheduler):
        return self._vision_tower.ema_update(momentum_scheduler)


    def requires_grad_(self, requires_grad):
        return self._vision_tower.requires_grad_(requires_grad)
