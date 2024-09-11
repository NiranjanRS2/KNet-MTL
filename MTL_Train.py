import torch
torch.__version__
import os, sys
import os.path as osp
import numpy as np
import os
import mmcv
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

''' TRAINING CHANGES '''

classes = ['Background', 'Dent']

mmsegmentation_path = '/media/inspektlabs/bot/Dent/package_copy/KNet-MTL/mmsegmentation'
sys.path.insert(0,mmsegmentation_path)

data_root = '/media/inspektlabs/bot/Dent/log39/Dataset'
LOG_DIR = '/media/inspektlabs/bot/Dent/log39'

img_dir = 'image'
ann_dir = 'mask'

config_file = os.path.join(mmsegmentation_path, "configs/knet/knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_ade20k.py")

IMG_SIZE=320
CROP_SIZE=320

PRE_TRAINED_PATH=None

RESUME_MODEL_PATH=None

maximum_iteration = 100000

checkpoint_save_interval = 12500

''' TRAINING CHANGES '''

split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    train_length = int(len(filename_list))   
    f.writelines(line + '\n' for line in filename_list)
    
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class DentDataset(CustomDataset):
    CLASSES = classes
    def __init__(self, split,**kwargs):
        try:
            del kwargs["times"]
            del kwargs["dataset"]
        except:
            pass
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

from mmcv import Config
from mmseg.apis import set_random_seed

cfg = Config.fromfile(config_file)

cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.device="cuda"

cfg.dataset_type = 'DentDataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 2

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(IMG_SIZE,IMG_SIZE), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(CROP_SIZE,CROP_SIZE), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(IMG_SIZE,IMG_SIZE),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.load_from = PRE_TRAINED_PATH
cfg.resume_from = RESUME_MODEL_PATH

cfg.work_dir = LOG_DIR

cfg.runner.max_iters = maximum_iteration
cfg.log_config.interval = 10
cfg.evaluation.interval = 100
cfg.checkpoint_config.interval = checkpoint_save_interval

print(f"Checkpoint will be saved on {cfg.checkpoint_config.interval} intervals")

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

datasets = [build_dataset(cfg.data.train)]

model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=None)

model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

from mmcv.runner import init_dist
from mmseg.utils import collect_env

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

setup(0, 1)

train_segmentor(model, datasets, cfg, distributed=False, validate=False, meta=dict())