_base_ =[
    './_base_/dataset256.py',
    './_base_/default_runtime.py',
    './_base_/scheduler50e.py'
]

custom_imports = dict(imports=['src'], allow_failed_imports=False)

# model settings
model = dict(
    type='ImageClassifier',
    pretrained = "https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-base-w12_3rdparty_in21k-192px_20220803-f7dc9763.pth",
    backbone=dict(
        type='SwinTransformerV2',
        arch='large',
        img_size=256,
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ArcFaceClsHead',
        num_classes=5000,
        in_channels=1536,
        loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)],),
)


if __name__ == '__main__':
    from mmcls.models import build_classifier
    import torch

    classifier = build_classifier(model)
    x = torch.rand( (1, 3, 256, 256) )
    y = classifier(inputs=x)
    print(y.size())



