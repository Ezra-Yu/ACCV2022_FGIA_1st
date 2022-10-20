_base_ =[
    './_base_/dataset384_b16.py',
    './_base_/default_runtime.py',
    './_base_/scheduler20e.py'
]

custom_imports = dict(imports=['src'], allow_failed_imports=False)

# model settings
model = dict(
    type='ImageClassifier',
    # pretrained = "https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-large-w24_in21k-pre_3rdparty_in1k-384px_20220803-3b36c165.pth",
    pretrained = "https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-large-w12_3rdparty_in21k-192px_20220803-d9073fee.pth",
    backbone=dict(
        type='SwinTransformerV2',
        arch='large',
        img_size=384,
        window_size=[24, 24, 24, 12],
        pretrained_window_sizes=[12, 12, 12, 6],
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5000,
        in_channels=1536,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8, num_classes=5000),
        dict(type='CutMix', alpha=1.0, num_classes=5000)
    ]),
)


if __name__ == '__main__':
    from mmcls.models import build_classifier
    import torch

    classifier = build_classifier(model)
    x = torch.rand( (1, 3, 384, 384) )
    y = classifier(inputs=x)
    print(y.size())




