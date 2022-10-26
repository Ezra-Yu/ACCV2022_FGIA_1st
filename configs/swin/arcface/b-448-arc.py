_base_ =[
    '../_base_/dataset448_b16.py',
    '../_base_/default_runtime.py',
    '../_base_/scheduler5e.py'
]

custom_imports = dict(imports=['src'], allow_failed_imports=False)

# model settings
model = dict(
    type='ImageClassifier',
    # pretrained = "",
    backbone=dict(
        type='SwinTransformerV2',
        arch='base',
        img_size=448,
        window_size=[28, 28, 28, 14],
        pretrained_window_sizes=[24, 24, 24, 12],
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ArcFaceClsHeadAdaptiveMargin',
        num_classes=5000,
        in_channels=1024,
        number_sub_center=3,
        # ann_file="./data/ACCV_workshop/meta/all.txt",
        loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)],),
)


if __name__ == "__main__":
    from mmcls.models import build_classifier
    import src
    import torch
    x = torch.rand( (1, 3, 448, 448) )

    cla = build_classifier(model)
    y = cla(x)
    print(y.size())