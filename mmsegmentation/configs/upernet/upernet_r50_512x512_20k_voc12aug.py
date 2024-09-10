_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=41), auxiliary_head=dict(num_classes=41))
