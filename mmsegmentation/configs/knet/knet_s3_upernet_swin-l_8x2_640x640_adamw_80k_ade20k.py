_base_ = 'knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k.py'

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'  # noqa
# model settings
model = dict(
    pretrained=checkpoint_file,
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        # drop_path_rate=0.4,
        drop_path_rate=0.2,
        patch_norm=True),
    decode_head=dict(
        kernel_generate_head=dict(in_channels=[192, 384, 768, 1536])),
    auxiliary_head=dict(in_channels=768))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (320,320) ## 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='Resize', img_scale=(320,320), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=(1080,1080), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320,320),
        # img_scale=(2048, 640),
        #         img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(
    _delete_=True,
    type='AdamW',
    # lr=0.000006,
    lr=0.000004,
    betas=(0.9, 0.999),
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
# optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    # target_ratio=(10, 1e-4),
    target_ratio=(5, 1e-5),
    cyclic_times=1,
    # step_ratio_up=0.4,
    step_ratio_up=0.3,
)
momentum_config = dict(
    policy='cyclic',
    # target_ratio=(0.85 / 0.95, 1),
    target_ratio=(0.9, 1),
    cyclic_times=1,
    # step_ratio_up=0.4,
    step_ratio_up=0.6,
)

# In K-Net implementation we use batch size 2 per GPU as default
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
