_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

N_CLASSES=4

model = dict(
    decode_head=dict(num_classes=N_CLASSES), auxiliary_head=dict(num_classes=N_CLASSES))
