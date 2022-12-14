_base_ =[
    './_base_/dataset384_b16.py',
    './_base_/default_runtime.py',
    './_base_/scheduler20e_arc.py'
]

_base_.train_dataloader.dataset.ann_file = "meta/rounda1/train.txt"

custom_imports = dict(imports=['src'], allow_failed_imports=False)

# model settings
model = dict(
    type='ImageClassifier',
    pretrained = "https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-large-w12_3rdparty_in21k-192px_20220803-d9073fee.pth",
    # pretrained = "https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-base-w12_3rdparty_in21k-192px_20220803-f7dc9763.pth",
    backbone=dict(
        type='SwinTransformerV2',
        arch='large',
        img_size=384,
        window_size=[24, 24, 24, 12],
        pretrained_window_sizes=[12, 12, 12, 6],
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ArcFaceClsHeadAdaptiveMargin',
        num_classes=5000,
        in_channels=1536,
        number_sub_center=3,
        ann_file="./data/ACCV_workshop/meta/rounda1/train.txt",
        loss = dict(type='SoftmaxEQLLoss', num_classes=5000),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)],),
)
