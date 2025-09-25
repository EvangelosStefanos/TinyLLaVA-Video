import math
import torch


class Multiblock3DMaskGenerator:
    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size[0], crop_size[1] // spatial_patch_size[1]
        self.duration = num_frames // temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))
        self.max_keep = max_keep

    def _sample_block_size(self, generator):
        # Temporal scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = self.temporal_pred_mask_scale
        t = max(1, int(self.duration * (min_t + _rand * (max_t - min_t))))

        # Spatial scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = self.spatial_pred_mask_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # Aspect ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = self.aspect_ratio
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, block_size, generator):
        t, h, w = block_size
        top = torch.randint(0, self.height - h + 1, (1,), generator=generator)
        left = torch.randint(0, self.width - w + 1, (1,), generator=generator)
        start = torch.randint(0, self.duration - t + 1, (1,), generator=generator)

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start:start + t, top:top + h, left:left + w] = 0

        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

        return mask

    def generate_masks(self, batch_size: int, seed=None, device=None):
        """
        Generate masks for already collated data.
        Returns:
            - mask_enc: LongTensor [B, N_visible]
            - mask_pred: LongTensor [B, N_masked]
        """
        if seed is None:
            seed = torch.seed()
        generator = torch.Generator()
        generator.manual_seed(seed)

        block_size = self._sample_block_size(generator)

        masks_enc, masks_pred = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width

        for _ in range(batch_size):
            while True:
                mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask *= self._sample_block_mask(block_size, generator)

                mask_e = torch.nonzero(mask.flatten(), as_tuple=False).squeeze()
                mask_p = torch.nonzero(1 - mask.flatten(), as_tuple=False).squeeze()

                if mask_e.numel() > 0:
                    min_keep_enc = min(min_keep_enc, mask_e.numel())
                    min_keep_pred = min(min_keep_pred, mask_p.numel())
                    masks_enc.append(mask_e)
                    masks_pred.append(mask_p)
                    break

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        masks_enc = [m[:min_keep_enc] for m in masks_enc]
        masks_pred = [m[:min_keep_pred] for m in masks_pred]

        masks_enc = torch.stack(masks_enc).to(device)
        masks_pred = torch.stack(masks_pred).to(device)

        return masks_enc, masks_pred


import numpy as np
from multiprocessing import Value


class RandomTubeMaskGenerator:
    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        self.mask_generators = [
            _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                ratio=cfg.get('ratio'),
            )
            for cfg in cfgs_mask
        ]

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, video_batch: torch.Tensor):
        """
        Args:
            video_batch: torch.Tensor of shape (B, C, T, H, W)

        Returns:
            masks_enc: list of Tensors of shape (B, N_visible)
            masks_pred: list of Tensors of shape (B, N_masked)
        """
        batch_size = video_batch.shape[0]

        all_masks_enc, all_masks_pred = [], []
        for mask_generator in self.mask_generators:
            masks_enc, masks_pred = mask_generator(batch_size)
            all_masks_enc.append(masks_enc)
            all_masks_pred.append(masks_pred)

        return all_masks_enc, all_masks_pred


class _MaskGenerator:
    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        ratio=0.9,
    ):
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2

        self.height = crop_size[0] // spatial_patch_size[0]
        self.width = crop_size[1] // spatial_patch_size[1]
        self.duration = num_frames // temporal_patch_size

        self.num_patches_spatial = self.height * self.width
        self.ratio = ratio
        self.num_keep_spatial = int(self.num_patches_spatial * (1. - self.ratio))
        self.num_keep = self.num_keep_spatial * self.duration

        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def __call__(self, batch_size):
        def sample_mask():
            mask = np.hstack([
                np.zeros(self.num_patches_spatial - self.num_keep_spatial),
                np.ones(self.num_keep_spatial),
            ])
            np.random.shuffle(mask)
            mask = torch.tensor(np.tile(mask, (self.duration, 1)))
            mask = mask.flatten()
            mask_pred = torch.nonzero(mask == 0).squeeze()
            mask_enc = torch.nonzero(mask == 1).squeeze()
            return mask_enc, mask_pred

        masks_enc, masks_pred = [], []
        for _ in range(batch_size):
            enc, pred = sample_mask()
            masks_enc.append(enc)
            masks_pred.append(pred)

        # Already collated batch; stack manually
        masks_enc = torch.stack(masks_enc, dim=0)
        masks_pred = torch.stack(masks_pred, dim=0)

        return masks_enc, masks_pred


def repeat_interleave_batch(x, B, repeat):
    """
    Repeat each sub-batch of size B 'repeat' times along the batch dimension.

    Args:
        x (Tensor): input tensor of shape (B * N, ...)
        B (int): sub-batch size
        repeat (int): number of repetitions per sub-batch

    Returns:
        Tensor: repeated tensor of shape (B * N * repeat, ...)
    """
    N = x.shape[0] // B
    x = x.view(N, B, *x.shape[1:])        # (N, B, ...)
    x = x.unsqueeze(1).expand(-1, repeat, -1, *([-1] * (x.ndim - 2)))  # (N, repeat, B, ...)
    x = x.reshape(N * repeat * B, *x.shape[3:])  # Flatten back to (N * repeat * B, ...)
    return x


