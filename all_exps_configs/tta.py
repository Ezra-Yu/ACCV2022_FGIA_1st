tta_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='TestTimeAug',
             transforms=[
                 [dict(type='Resize', scale=256, backend='pillow')],
                 [dict(type='RandomFlip', prob=1.), dict(type='RandomFlip', prob=0.)],
                 [dict(type='LoadAnnotations', with_bbox=True)],
                 [dict(type='PackDetInputs')],
             ]
        )
]