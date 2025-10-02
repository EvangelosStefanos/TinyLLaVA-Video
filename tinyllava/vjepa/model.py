import torch
import torch.nn.functional as F
import copy

import jepa.src.models.vision_transformer as ViT
import jepa.src.models.predictor as ViTPredictor 
from jepa.src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from jepa.src.utils.tensors import trunc_normal_
from jepa.src.masks.utils import apply_masks
from tinyllava.vjepa.masks import repeat_interleave_batch, MultiMaskGenerator


def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
):
    encoder = ViT.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    encoder = MultiMaskWrapper(encoder)
    predictor = ViTPredictor.__dict__['vit_predictor'](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
    )
    predictor = PredictorMultiMaskWrapper(predictor)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to('cpu')
    predictor.to('cpu')
    print(encoder)
    print(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Encoder number of parameters: {count_parameters(encoder)}')
    print(f'Predictor number of parameters: {count_parameters(predictor)}')

    return encoder, predictor


class VJEPAModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder, self.predictor = init_video_model(
            uniform_power=cfg['uniform_power'],
            use_mask_tokens=cfg['use_mask_tokens'],
            num_mask_tokens=len(cfg['cfgs_mask']),
            zero_init_mask_tokens=cfg['zero_init_mask_tokens'],
            device=torch.device('cpu'),
            patch_size=cfg['patch_size'],
            num_frames=cfg['num_frames'],
            tubelet_size=cfg['tubelet_size'],
            model_name=cfg['model_name'],
            crop_size=cfg['crop_size'],
            pred_depth=cfg['pred_depth'],
            pred_embed_dim=cfg['pred_embed_dim'],
            use_sdpa=cfg['use_sdpa'],
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)

        self.mask_generators = MultiMaskGenerator(cfg)
        self.cfg = cfg
        return


    def load_model(self, vision_tower_name, **kwargs):
        try:
            checkpoint = torch.load(vision_tower_name, map_location=torch.device('cpu'))
        except Exception as e:
            print(f'Encountered exception when loading checkpoint {e}')

        epoch = 0
        try:
            epoch = checkpoint['epoch']

            # -- loading encoder
            pretrained_dict = checkpoint['encoder']
            msg = self.encoder.load_state_dict(pretrained_dict)
            print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

            # -- loading predictor
            pretrained_dict = checkpoint['predictor']
            msg = self.predictor.load_state_dict(pretrained_dict)
            print(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

            # -- loading target_encoder
            if self.target_encoder is not None:
                print(list(checkpoint.keys()))
                pretrained_dict = checkpoint['target_encoder']
                msg = self.target_encoder.load_state_dict(pretrained_dict)
                print(
                    f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}'
                )

            del checkpoint

        except Exception as e:
            print(f'Encountered exception when loading checkpoint {e}')
            epoch = 0
        return


    def forward(self, x, **kwargs):
        # x: frames of shape [T, C, H, W]
        # return: tensor of shape [B, N, D] a compressed representation of tokens
        x = x.unsqueeze(0)
        B, T, C, H, W = x.shape
        
        # T must be divisible by num_clips # TODO: [low priority] pad the video if not divisible.
        assert T % self.cfg['num_clips'] == 0, f"Input video length {T} not compatible \
            with num_clips {self.cfg['num_clips']} and num_frames {self.cfg['num_frames']}"
        x = x.permute((0, 2, 1, 3, 4)).view(B * self.cfg['num_clips'], C, -1, H, W) # split into multiple clips if num_clips > 1
        
        masks_enc, masks_pred = self.mask_generators(B) # TODO: Test to ensure that masking is done correctly.

        # use the same mask for all clips of the same video
        masks_enc = masks_enc[::self.cfg['num_clips']]
        masks_pred = masks_pred[::self.cfg['num_clips']]
        
        for i in range(len(masks_enc)):
            masks_enc[i] = masks_enc[i].to(x.device)
            masks_pred[i] = masks_pred[i].to(x.device)

        masks_enc = [repeat_interleave_batch(m, B, self.cfg['num_clips']) for m in masks_enc]
        masks_pred = [repeat_interleave_batch(m, B, self.cfg['num_clips']) for m in masks_pred]
        
        # Forward target encoder (no grad)
        with torch.no_grad():
            h = self.target_encoder(x)
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, masks_pred, concat=False)

        # Forward encoder and predictor
        z = self.encoder(x, masks_enc) # list (len=num_masks) of tensors of shape [B * num_clips, num_unmasked_tokens, D]
        z = self.predictor(z, h, masks_enc, masks_pred) # list (len=num_masks) of tensors of shape [B * num_clips, num_masked_tokens, D]

        self.loss = self.loss_fn(z, h, masks_pred)
        return z[0] # TODO: return only one masked prediction in case of multi-mask training.


    def loss_fn(self, z, h, masks_pred):
        # Loss
        loss_jepa = 0.
        for zi, hi in zip(z, h):
            loss_jepa += torch.mean(torch.abs(zi - hi)**self.cfg['loss_exp']) / self.cfg['loss_exp']
        loss_jepa /= len(masks_pred)

        # Regularization
        pstd_z = sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
        loss_reg = torch.mean(F.relu(1.-pstd_z))
        
        return loss_jepa + self.cfg['reg_coeff'] * loss_reg
    
    
    def get_loss(self):
        return self.loss
    
    
    def ema_update(self, momentum_scheduler):
        m = next(momentum_scheduler)
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        return
    
    
    def requires_grad_(self, requires_grad):
        self.encoder.requires_grad_(requires_grad)
        self.predictor.requires_grad_(requires_grad)
        return

