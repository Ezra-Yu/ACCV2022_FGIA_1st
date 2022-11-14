# 复现精度


## 准备

### 下载代码

```shell
git clone https://github.com/Ezra-Yu/ACCV_workshop.git
git checkout dev
mkdir data
mkdir data/ACCV_workshop
mkdir checkpoints
```

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
├── pkls                             # 存放所有的推理间接结果
├── src                              # 所有源代码
├── tools                            # 所有训练测试以及各种工具脚本
├── docker/Dockerfile                # docker镜像
├── docker-compose.yml
├── requirements.txt
└── READ.md
```


### 准备数据集

1. 下载所有的数据集并解压为 train, testa, testb 文件夹，放如`data/ACCV_workshop/` 文件夹下并进入；

    ```shell
    cd data/ACCV_workshop/
    ```

3. 在 train 文件夹下建立指向 testa, testb 的**软链接**

    ```shell
    ln -s ./testa train/testa
    ln -s ./testb train/testb
    ```

3. 下载所有的 meta 文件并解压;
    
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
    │   ├── rounda1/train.txt      # 加入testa的pseudo，第1轮训练集标注
    │   ├── rounda2/train.txt      # 加入testa的pseudo，第2轮训练集标注
    │   ├── rounda3/train.txt      # 加入testa的pseudo，第3轮训练集标注
    │   ├── roundb1/train.txt      # 加入testa以及testb的pseudo，第1轮训练集标注
    │   ├── roundb2/train.txt      # 加入testa以及testb的pseudo，第2轮训练集标注
    │   └── roundb3/train.txt      # 加入testa以及testb的pseudo，第3轮训练集标注
```


### 启动容器

设置环境变量, 在项目路径下：

```shell
export DATA_DIR="./data"   # 
export PYTHONPATH=`pwd`:$PYTHONPATH
```

启动一个容器，在项目目录运行：

```shell
docker-compose up -d accv
```

出现：

```
xxx.....
Creating accv ... done
```


## 推理

下载checkpoints

```shell
wget -O  "swin-l-384-arc-roundb3-eql-7560.pth" https://tmp-titan.vx-cdn.com/file-6371fc6702379-6372029c1559e/swin-l-384-arc-roundb3-eql-7560.pth ./checlpoints/
```

#### 单机单卡

```shell
docker-compose exec accv python tools/infer_folder.py configs/swin/l-384-arc_roundb3.py ./checkpoints/swin-l-384-arc-roundb3-eql-7560.pth ./demo/ --dump pkls/swin-l-384-arc-roundb3-eql-7560.pkl --tta --cfg-option test_dataloader.batch_size=1
```

#### 单机多卡

```shell
docker-compose exec accv python tools/infer_folder.py configs/swin/l-384-arc_roundb3.py ./swin-l-384-arc-roundb3-eql-7560.pth ./demo/ --dump pkls/swin-l-384-arc-roundb3-eql-7560.pkl --tta --cfg-option test_dataloader.batch_size=1
```

### 下载 checkpoints



## 训练

### ViT

### Swin

**Single GPUS**

**Multi GPUS**

**Slurm**


### Create Pseudo Label



