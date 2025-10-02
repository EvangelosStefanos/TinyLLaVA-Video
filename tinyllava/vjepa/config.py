import yaml
import torch


_GLOBAL_SEED = 0


class VJEPAConfig:
    def __init__(self):
        fname = 'jepa/configs/pretrain/vitl16.yaml'
        # Load config
        params = None
        with open(fname, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            print(f'Loading configuration from "{fname}".')
        
        args = params
        resume_preempt = False

        # -- META
        cfgs_meta = args.get('meta')
        self.load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
        self.r_file = cfgs_meta.get('read_checkpoint', None)
        self.seed = cfgs_meta.get('seed', _GLOBAL_SEED)
        self.save_every_freq = cfgs_meta.get('save_every_freq', -1)
        self.skip_batches = cfgs_meta.get('skip_batches', -1)
        self.use_sdpa = cfgs_meta.get('use_sdpa', False)
        self.which_dtype = cfgs_meta.get('dtype')
        # print(f'{self.which_dtype=}')
        # if self.which_dtype.lower() == 'bfloat16':
        #     self.dtype = torch.bfloat16
        #     self.mixed_precision = True
        # elif self.which_dtype.lower() == 'float16':
        #     self.dtype = torch.float16
        #     self.mixed_precision = True
        # else:
        #     self.dtype = torch.float32
        #     self.mixed_precision = False

        # -- MASK
        self.cfgs_mask = args.get('mask')

        # -- MODEL
        cfgs_model = args.get('model')
        self.model_name = cfgs_model.get('model_name')
        self.pred_depth = cfgs_model.get('pred_depth')
        self.pred_embed_dim = cfgs_model.get('pred_embed_dim')
        self.uniform_power = cfgs_model.get('uniform_power', True)
        self.use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
        self.zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

        # -- DATA
        cfgs_data = args.get('data')
        self.dataset_type = cfgs_data.get('dataset_type', 'videodataset')
        self.mask_type = cfgs_data.get('mask_type', 'multiblock3d')
        self.dataset_paths = cfgs_data.get('datasets', [])
        self.datasets_weights = cfgs_data.get('datasets_weights', None)
        if self.datasets_weights is not None:
            assert len(self.datasets_weights) == len(self.dataset_paths), 'Must have one sampling weight specified for each dataset'
        self.batch_size = cfgs_data.get('batch_size')
        self.num_clips = cfgs_data.get('num_clips')
        self.num_frames = cfgs_data.get('num_frames')
        self.frames_per_clip = self.num_frames // self.num_clips
        self.tubelet_size = cfgs_data.get('tubelet_size')
        self.sampling_rate = cfgs_data.get('sampling_rate')
        self.duration = cfgs_data.get('clip_duration', None)
        self.crop_size = cfgs_data.get('crop_size', 224)
        self.patch_size = cfgs_data.get('patch_size')
        self.pin_mem = cfgs_data.get('pin_mem', False)
        self.num_workers = cfgs_data.get('num_workers', 1)
        self.filter_short_videos = cfgs_data.get('filter_short_videos', False)
        self.decode_one_clip = cfgs_data.get('decode_one_clip', True)
        self.log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

        # -- DATA AUGS
        cfgs_data_aug = args.get('data_aug')
        self.ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
        self.rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
        self.motion_shift = cfgs_data_aug.get('motion_shift', False)
        self.reprob = cfgs_data_aug.get('reprob', 0.)
        self.use_aa = cfgs_data_aug.get('auto_augment', False)

        # -- LOSS
        cfgs_loss = args.get('loss')
        self.loss_exp = cfgs_loss.get('loss_exp')
        self.reg_coeff = cfgs_loss.get('reg_coeff')

        # -- OPTIMIZATION
        cfgs_opt = args.get('optimization')
        self.ipe = cfgs_opt.get('ipe', None)
        self.ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
        self.clip_grad = cfgs_opt.get('clip_grad', None)
        self.wd = float(cfgs_opt.get('weight_decay'))
        self.final_wd = float(cfgs_opt.get('final_weight_decay'))
        self.num_epochs = cfgs_opt.get('epochs')
        self.warmup = cfgs_opt.get('warmup')
        self.start_lr = cfgs_opt.get('start_lr')
        self.lr = cfgs_opt.get('lr')
        self.final_lr = cfgs_opt.get('final_lr')
        self.ema = cfgs_opt.get('ema')
        self.betas = cfgs_opt.get('betas', [0.9, 0.999])
        self.eps = cfgs_opt.get('eps', 1.e-8)

        # -- LOGGING
        cfgs_logging = args.get('logging')
        self.folder = cfgs_logging.get('folder')
        self.tag = cfgs_logging.get('write_tag')
        return


def get_vjepa_config():
    return {'vjepa': VJEPAConfig().__dict__}
