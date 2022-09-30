model = dict(
    type='ImageClassifier',
    backbone=dict(type='TimmEfficientNet',
                model_name='tf_efficientnet_l2_ns',
                drop_path_rate=0.2,
                pretrained=True,
                with_cp=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ArcFaceHead',
        num_classes=5000,
        in_channels=5504,
        feature_norm=True,
        weight_norm=True,
        loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        init_cfg=dict(type='Normal', layer='Linear', std=0.01))
)