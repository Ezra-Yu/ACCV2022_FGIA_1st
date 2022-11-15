# 复现精度


## 准备

### 解压代码


项目结构：

```
$ACCV_workshop
├── config                         # 所有的 config
│   ├── swin                       # 所有的 swin 相关config
│   │   ├── _base_                 # 基础配置
│   │   ├── b-384-arc_roundb1.py    
│   │   └── ......
│   └── vit                         # 所有的 vit 相关config
|        ├── _base_                 # 基础配置
|        ├── b-384-arc_roundb1.py    
|        └── ......
├── data                             # 数据集
├── checkpoints                      # 存放所有的权重
├── testb-pkls                       # 存放所有的推理间接结果
├── src                              # 所有源代码
├── tools                            # 所有训练测试以及各种工具脚本
├── docker/Dockerfile                # docker镜像
├── docker-compose.yml
├── requirements.txt
└── README.md
```


### 准备数据集

1. 下载所有的数据集并解压为 train, testa, testb 文件夹，放入`data/ACCV_workshop/` 文件夹下并进入；

3. 在 train 文件夹下建立指向 testa, testb 的**软链接**

    ```shell
    cd data/ACCV_workshop/
    ln -s ./testa train/testa
    ln -s ./testb train/testb
    ```

3. 下载所有的 meta 文件并解压,如果存在，不需要处理;
    
    ```shell
    wget -O  "meta.zip" https://tmp-titan.vx-cdn.com/file-6371f3c54bfb1-6371f4801d3c4/meta.zip 
    unzip meta.zip
    ```

最后数据集的目录结构为: 

```shell
data/ACCV_workshop
    ├── train
    │   ├── testa   # 指向 testa 的**软链接**, soft link
    │   ├── testb   # 指向 testb 的**软链接**, soft link
    │   ├── 0000 
    │   ├── 0001    
    │   └── ...... 
    ├── testa
    │   ├── xxxxxx.jpg
    │   ├── yyyyyy.jpg  
    │   └── ...... 
    ├── testb
    │   ├── 1111111.jpg
    │   ├── 222222.jpg  
    │   └── ...... 
    ├── meta
    │   ├── train.txt              # 数据清洗后的训练集标注
    │   ├── all.txt                # 数据清洗前的训练集标注
    │   ├── rounda1/train.txt      # 加入testa的pseudo，第1轮训练集标注
    │   ├── rounda2/train.txt      # 加入testa的pseudo，第2轮训练集标注
    │   ├── rounda3/train.txt      # 加入testa的pseudo，第3轮训练集标注
    │   ├── roundb1/train.txt      # 加入testa以及testb的pseudo，第1轮训练集标注
    │   ├── roundb2/train.txt      # 加入testa以及testb的pseudo，第2轮训练集标注
    │   └── roundb3/train.txt      # 加入testa以及testb的pseudo，第3轮训练集标注
```


### 启动

在 ACCV_workshop 项目根目录下。

构建镜像：

```
sudo docker build -t openmmlab:accv docker/
```

启动容器：

```
sudo docker run -it \
    -v /home/PJLAB/yuzhaohui/repos/ACCV_workshop:/workspace/ACCV_workshop \
    -w /workspace/ACCV_workshop \
    -e PYTHONPATH=/working:$PYTHONPATH \
    openmmlab:accv  /bin/bash 
```

也可以使用 docker-compose， (以下的命令根据docker-compose修改)

```shell
docker-compose up -d accv
```

## 集成以及re-distribute-label (快速得到结果)

pkl的目录: 

```shell
testb-pkls (17个pkl)
    ├── swin-b-384-arc-roundb1-testb-7410.pkl         # 需要在最后加上精度
    ├── swin-b-384-arc-roundb2-testb-7460.pkl
    ├── mae-vit-ce-30e_semi-3st-thres4-7471.pkl        
    ├── ....
    └── vit-448-arc-30e-testb-3st-7620.pkl     
```

**注意**： roundb 的 pkl 精度都是预估出来，每次在之前基础上提高0.5， roundb1 的结果都是a榜提交结果

#### 集成

```
python tools/emsemble.py --pkls-dir testb-pkls --factor 25 --scale --dump-all testb.pkl
```
看到：
```text
Number of .pkls is 17....
Adjusted factor is: [2.080085996252569, 1.3987451430721878, 1.2237077021500773, 1.0626193781723274, 1.1833516629706529, 1.0, 1.764428216037884, 1.0343373472253807, 1.446123660572264, 1.7070722173927184, 1.2781492311812328, 1.269624858196828, 1.017029565305733, 1.6515084398827093, 1.2237077021500773, 1.446123660572264, 1.495040719791741]
```

可以 ```zip pred_results.zip pred_results.csv``` 打包提交得到 **0.7815**, 'testb.pkl' 是临时中间结果，给下面使用。

#### re-distribute-label

```
python tools/re-distribute-label.py testb.pkl --K 16
```

可以 ```zip pred_results.zip pred_results.csv``` 打包提交得到 **0.7894**,

## 推理

#### 单机单卡

```shell
python tools/infer_folder.py configs/swin/l-384-arc_roundb3.py checkpoints/swin-l-384-arc-roundb3-7560.pth ./data/ACCV_workshop/testb --dump pkls/swin-l-384-arc-roundb3-7560.pkl --tta --cfg-option test_dataloader.batch_size=32
```

#### 单机多卡

```shell
python -m torch.distributed.launch --nproc_per_node=4  tools/infer_folder.py configs/swin/l-384-arc_roundb3.py checkpoints/swin-l-384-arc-roundb3-7560.pth ./data/ACCV_workshop/testb --dump pkls/swin-l-384-arc-roundb3-7560.pkl --tta --cfg-option test_dataloader.batch_size=32 --launcher pytorch
```

**test_dataloader.batch_size=32** batch_size 根据需要修改


## 训练

### ViT

### Swin

swin-b 需要 8张卡， swin-l 需要 16 张卡；

**Multi GPUS**

```
python -m torch.distributed.launch --nproc_per_node=16  tools/train.py configs/swin/l-384-arc_roundb3.py ~/accv/l-384-arc_roundb3  --amp
```

**Slurm**

```
GPUS=16 sh ./tools/slurm_train.sh ${CLUSTER_NAME} ${JOB_NAME} configs/swin/l-384-arc_roundb3.py ~/accv/l-384-arc_roundb3 --amp
```

### Uniform Model Soup

融合训练得到的模型， swin-b 融合最后 5个， swin-l 融合 最后的 7 个。将需要的checkpoint 放在一个文件夹中。使用以下命令

```
python tools/model_soup.py --model-folder ${CKPT-DIR} --out ${Final-CKPT}
```

### Create Pseudo Label

Get needed inter-result

```shell
python tools/emsemble.py --pkls-dir testb-pkls --dump testb-pseudo.pkl  # 不要加 --scale --factor 25
python tools/creat_pseudo.py testb-pseudo.pkl --thr 0.45 --testb   
```


可以得到：
```text
90000 samples have been found....
Get 78458 pseudo samples....
```

注意：
1. 第一步不要'--scale --factor 25'
2. 第二部需要根据当前的数据集， ``--testb``表示生成的 'testb' 的标签， 不加为 'testa'的标签；
   区别为生成的 'pseudo.txt' 中的图片路径前缀不同，分别为 'testb' 和 'testa'
3. 生成的 'pseudo.txt' 需要和之前的训练标注合并起来才能使用
4. 想使用必须在 './data/ACCV_workshop/train' 建立 'testa' 与 'testb' 的软连接。