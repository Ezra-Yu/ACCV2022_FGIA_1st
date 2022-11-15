# ACCA_workshop
For the competition in https://www.cvmart.net/race/10412/base


Mainly base on [**MMClassifiion**](https://github.com/open-mmlab/mmclassification) 与 [**MMSelfSup**](https://github.com/open-mmlab/mmselfsup). Wlcome to use them and star, flork.

## Result / 结果

** LB A**

![A](https://user-images.githubusercontent.com/18586273/201299318-41d392b3-a810-4f26-bd29-6dabba92fe83.png)

**LB B**

![B](https://user-images.githubusercontent.com/18586273/201299402-d4449a50-a48a-46e6-b673-049524d81bdb.png)


## Reproduce / 复现

**复现精度请点击[这里](./Reproduce.md)**

## 算法描述

### 算法选择

- ViT(MAE-pt)   # 自己预训练，pt-16A100-1周，
- Swin(21kpt)    

**主要结构**
- ViT + CE-loss + post-LongTail-Adjusment                # ft-16A100-18h   
- ViT + SubCenterArcFaceWithAdvMargin(CE)                # ft-16A100-18h
- Swin-B + SubCenterArcFaceWithAdvMargin(Soft-EQL)       # ft-8A100-16h
- Swin-L + SubCenterArcFaceWithAdvMargin(Soft-EQL)       # ft-16A100-13h

所有都使用了 **Flip TTA**。

## 复现步骤

预训练 --> 训练 --> 清洗数据 --> 训练与生成伪标签交替 --> 模型集成 --> 调整预测分布

1. MAE 预训练
2. Swin 与 ViT 的训练
3. 数据清洗         
4. Swin 与 ViT 的训练 rounda1
5. 制作伪标签testa1
6. Swin 与 ViT 的训练 rounda2
7. 使用新的CKPT继续做伪标签testa2
8. Swin 与 ViT 的训练 rounda3
9. 提交testb，做伪标签testb1
10. Swin 与 ViT 的训练 roundb1
11. 提交testb，做伪标签testb2
12. Swin 与 ViT 的训练 roundb1
13. 模型融合，调整预测的标签分布，提交


## 技术总结 tricks summary

- [MAE](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mae) |  [Config](./configs/vit/)    | best 7460@A
- [Swinv2](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/swin_transformer_v2) | [Config](./configs/swin/)  | best 7400@A
- [ArcFace](https://arxiv.org/abs/1801.07698)   |   [Code](./src/models/arcface_head.py)   about + 0.2 (swin)
- [SubCenterArcFaceWithAdvMargin](https://paperswithcode.com/paper/sub-center-arcface-boosting-face-recognition)   |   [Code](./src/models/arcface_head.py) about + 0.3 (swin)
- [Post-LT-adjusment](https://paperswithcode.com/paper/long-tail-learning-via-logit-adjustment)   |   [Code](./src/models/linear_head_lt.py)  about + 0.4 (CE)
- [SoftMaxEQL](https://paperswithcode.com/paper/the-equalization-losses-gradient-driven)   |   [Code](./src/models/eql.py)   about + 0.6 (Swin)
- FlipTTA  |   [Code](./src/models/tta_classifier.py)    about + 0.2
- 数据清洗                                                about + 0.5
- 模型融合: [Uniform-model-soup](https://arxiv.org/abs/2203.05482) | [code](./tools/model_soup.py)             about +0.4 for Swin
- [半监督](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)  | [Code](./tools/creat_pseudo.py)  about +3 for all
- 自适应集成 [Code](./tools/emsemble.py),                  FROM 7634@A -> 7642@A， 直接集成 7634
- 后处理: [调整预测分布](./tools/re-distribute-label.py);    FROM 7642@A -> 7732@A 
- 

#### 使用了没有效果 used but no improvement

1. 使用检索以及检索分类混合的方式
2. 使用EfficientNet

#### 后续 not used but worth try

1. DiVE 蒸馏提升长尾问题表现
2. Simim 训一个 swinv2 的预训练模型
3. 优化 re-distribute-label

## 参考



