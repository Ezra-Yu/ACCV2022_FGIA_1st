_base_ =[
    './_base_/dataset448.py',
    './_base_/default_runtime.py',
    './_base_/scheduler20e.py'
]

custom_imports = dict(imports=['src', 'mmselfsup.engine' ], allow_failed_imports=False)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='large',
        img_size=448,
        patch_size=16,
        drop_path_rate=0.1,
        avg_token=True,
        output_cls_token=False,
        final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='~/accv/liuyuan/mae_webnat_pretrain/epoch_1600.pth',
            prefix='backbone.')),
    neck=None,
    head=dict(
        type='ArcFaceClsHeadAdaptiveMargin',
        num_classes=5000,
        in_channels=1024,
        number_sub_center=3,
        ann_file="./data/ACCV_workshop/meta/all.txt",
        loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)],),
)
