_base_ =[
    './_base_/dataset256.py',
    './_base_/default_runtime.py',
    './_base_/scheduler20e_arc.py'
]

custom_imports = dict(imports=['src'], allow_failed_imports=False)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TIMMBackbone',
        model_name='maxxvit_rmlp_small_rw_256',
        # features_only=True,
        pretrained=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ArcFaceClsHeadAdaptiveMargin',
        num_classes=5000,
        in_channels=768,
        number_sub_center=3,
        ann_file="./data/ACCV_workshop/meta/all.txt",
        loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)],),
)

if __name__ == "__main__":
    from mmcls.registry import MODELS
    import src

    import torch
    x = torch.rand( (1, 3, 256, 256) )

    m = MODELS.build(model)
    y = m(x)
    print(y.size())
