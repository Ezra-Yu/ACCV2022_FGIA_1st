# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
train_pipeline = [
    dict(type='LoadImageFromFile',
         file_client_args=dict(
            backend='memcached',
            server_list_cfg='/mnt/lustre/share/pymc/pcs_server_list.conf',
            client_cfg='/mnt/lustre/share/pymc/mc.conf',
            sys_path='/mnt/lustre/share/pymc')),
    dict(type='Resize', scale=768),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean])),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile',
        file_client_args=dict(
            backend='memcached',
            server_list_cfg='/mnt/lustre/share/pymc/pcs_server_list.conf',
            client_cfg='/mnt/lustre/share/pymc/mc.conf',
            sys_path='/mnt/lustre/share/pymc')),
    dict(type='Resize', scale=768),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/ACCV_workshop',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/ACCV_workshop',
        ann_file='meta/val.txt',
        data_prefix='train',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/ACCV_workshop',
        ann_file='meta/test.txt',
        data_prefix='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)