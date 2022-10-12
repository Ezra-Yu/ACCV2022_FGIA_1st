# ACCA_workshop
For the competition in https://www.cvmart.net/race/10412/base


# Prepare

install mmcv, mmengine, mmselfsup, mmcls

```
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
pip install "mmcls>=1.0.0rc1"
pip install 'mmselfsup>=1.0.0rc1'
```

# How to inference 

### Local

```
python -u tools/infer_folder.py ${CONFIG} ${CHECKPOINT} ${FOLDER} --out-keys filename pred_class --out pred_results.csv
```

### Slurm

```
GPUS=1 GPUS_PER_NODE=1 sh ./tools/slurm_infer.sh mm_model accv_test ${CONFIG} ${CHECKPOINT} ${TEST_IMAGE_FOLDER}
```

output `pred_results.csv` and `pred_results.zip`, just submit `pred_results.zip`.

### TTA
```
GPUS=1 GPUS_PER_NODE=1 sh ./tools/slurm_infer.sh mm_model accv_test ${CONFIG} ${CHECKPOINT} ${TEST_IMAGE_FOLDER} --tta
```

### DUMP

dump all the result, including scores to do emsemble

```
GPUS=1 GPUS_PER_NODE=1 sh ./tools/slurm_infer.sh mm_model accv_test ${CONFIG} ${CHECKPOINT} ${TEST_IMAGE_FOLDER} --tta --dump ${DUMP_PATH}
```

### Check

You can infernece a train subfolder to check model validity, the filename contains the label infomation, like

```
GPUS=1 GPUS_PER_NODE=1 sh ./tools/slurm_infer.sh mm_model accv_test configs/liuyuan.py ~/accv/liuyuan/mae_webnat_1600e_pt_50e_ft_ckpt/epoch_50.pth ./data/ACCV_workshop/train/0002/
```

the result will be like :

```
 0002_0fcdce76ec165d97b61bb2463355f05df8287775.jpg,0002         
 0002_c282d27f595010ec1b04bd2d79d7fa280598ed74.jpg,0002       
 0002_e9f23d9ab4217d8322a7521712d8bef0464cc031.jpg,0002
```

# How to train and test

**Remember to change the test dataset to val**

detial refer to [MMCLS DOC](https://mmclassification.readthedocs.io/en/1.x/user_guides/train_test.html#training)

