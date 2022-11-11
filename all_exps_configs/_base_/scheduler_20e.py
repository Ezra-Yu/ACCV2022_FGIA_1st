# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0,))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=0.001 * 0.01,
        by_epoch=True,
        begin=1,
        end=16,
    ),
    dict(
        type='LinearLR',
        start_factor = 1,
        end_factor = 1e-2,
        by_epoch=True,
        begin=16,
        end=20,
        convert_to_iter_based=True),
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=64 * 8)
