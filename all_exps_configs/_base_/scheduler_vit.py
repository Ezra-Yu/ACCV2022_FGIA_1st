
custom_imports = dict(imports=['mmselfsup.engine'], allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.004,
        weight_decay=0.05,
        model_type='vit',
        eps=1e-08,
        betas=(0.9, 0.999),
        layer_decay_rate=0.75,
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })),
   constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor'
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        by_epoch=True,
        begin=5,
        end=50,
        eta_min=1e-06,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=1024)
