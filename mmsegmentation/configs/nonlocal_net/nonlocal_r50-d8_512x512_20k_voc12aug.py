_base_ = [
    '../_base_/models/nonlocal_r50-d8.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=31), auxiliary_head=dict(num_classes=31))
