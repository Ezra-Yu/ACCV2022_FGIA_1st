_base_ =[
    './_base_/dataset_384.py',
    './_base_/default_runtime.py',
    './_base_/scheduler_20e.py'
]

custom_imports = dict(imports=['src'], allow_failed_imports=False)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='TimmEfficientNet',
                model_name = 'tf_efficientnetv2_s_in21k',
                pretrained=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5000,
        in_channels=1280,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-05)]),
    train_cfg=dict(
        augments=[
            dict(type='Mixup', alpha=0.8),
            dict(type='CutMix', alpha=1.0)]),
)
