_base_ =[
    './_base_/dataset.py',
    './_base_/default_runtime.py',
    # './_base_/efficientnet.py',
    './_base_/scheduler.py'
]

custom_imports = dict(imports=['src'], allow_failed_imports=False)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='TimmEfficientNet',
                # model_name='tf_efficientnet_l2_ns',
                model_name = 'tf_efficientnetv2_l_in21ft1k',
                drop_path_rate=0.2,
                pretrained=True,
                with_cp=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ArcFaceHead',
        num_classes=5000,
        # in_channels=5504,
        in_channels=1280,
        feature_norm=True,
        weight_norm=True,
        used='after',
        loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='Normal', layer='Linear', std=0.01)),
)

