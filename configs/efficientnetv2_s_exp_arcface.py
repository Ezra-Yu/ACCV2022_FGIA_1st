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
        type='ArcFaceClsHead',
        num_classes=5000,
        in_channels=1280,
        loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='Normal', layer='Linear', std=0.01)),
)
