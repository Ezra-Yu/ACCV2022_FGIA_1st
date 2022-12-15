# Solution of FGIA ACCV 2022(1st Place)

This is the 1st Place Solution for Webly-supervised Fine-grained Recognition, refer to the ACCV workshop competition in https://www.cvmart.net/race/10412/base.

Mainly done by [Ezra-Yu](https://github.com/Ezra-Yu), [Yuan Liu](https://github.com/YuanLiuuuuuu) and [Songyang Zhang](https://github.com/tonysy), base on [**MMClassifiion**](https://github.com/open-mmlab/mmclassification) 与 [**MMSelfSup**](https://github.com/open-mmlab/mmselfsup). please flork and star them if you think they are useful.

## Result

<details>

<summary>Show the result</summary>

<br>

**LB A**

![LB-A](https://user-images.githubusercontent.com/18586273/205498131-5728e470-b4f6-43b7-82a5-5f8e3bd5168e.png)

**LB B**

![LB-B](https://user-images.githubusercontent.com/18586273/205498171-5a3a3055-370a-4a8b-9779-b686254ebc94.png)

</br>

</details>


## Reproduce / 复现

**复现精度请点击[这里](./Reproduce.md)**

## Description

### Backbone and Pre-train

- ViT(MAE-pt)   # 自己预训练
- Swin(21kpt)   # 来自[MMCls-swin_transformer_v2](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/swin_transformer_v2).

**Archs**
- ViT + CE-loss + post-LongTail-Adjusment                 
- ViT + SubCenterArcFaceWithAdvMargin(CE)              
- Swin-B + SubCenterArcFaceWithAdvMargin(SoftMax-EQL)  
- Swin-L + SubCenterArcFaceWithAdvMargin(SoftMAx-EQL) 

所有都使用了 **Flip TTA**。

## Flow

MIM 预训练 --> 训练 --> 清洗数据 --> fine-tune + 集成 + 生成伪标签交替训练 --> 后处理

1. MAE 预训练
2. Swin 与 ViT 的训练
3. 使用权重做数据清洗         
4. 训练 -> 制作伪标签，放回训练集中 -> 再训练； (testa3轮)
5. 训练 -> 制作伪标签，放回训练集中 -> 再训练； (testb3轮,包括testa的伪标签)
6. 模型融合，调整预测的标签分布，提交

![flow](https://user-images.githubusercontent.com/18586273/205498371-31dbc1f4-5814-44bc-904a-f0d32515c7dd.png)

## Summary

- [MAE](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mae) |  [Config](./configs/vit/)  
- [Swinv2](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/swin_transformer_v2) | [Config](./configs/swin/) 
- [ArcFace](https://arxiv.org/abs/1801.07698)   |   [Code](./src/models/arcface_head.py)  
- [SubCenterArcFaceWithAdvMargin](https://paperswithcode.com/paper/sub-center-arcface-boosting-face-recognition)   |   [Code](./src/models/arcface_head.py) 
- [Post-LT-adjusment](https://paperswithcode.com/paper/long-tail-learning-via-logit-adjustment)   |   [Code](./src/models/linear_head_lt.py) 
- [SoftMaxEQL](https://paperswithcode.com/paper/the-equalization-losses-gradient-driven)   |   [Code](./src/models/eql.py)   
- FlipTTA  |   [Code](./src/models/tta_classifier.py)   
- 数据清洗                                                
- 模型融合: [Uniform-model-soup](https://arxiv.org/abs/2203.05482) | [code](./tools/model_soup.py)            
- [半监督](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)  | [Code](./tools/creat_pseudo.py)  
- 自适应集成 [Code](./tools/emsemble.py),                  
- 后处理: [调整预测分布](./tools/re-distribute-label.py);    


![image](https://user-images.githubusercontent.com/18586273/205498027-def99b0d-a99a-470b-b292-8d5fc83111fc.png)

#### 使用了没有效果 used but no improvement

1. 使用检索以及检索分类混合的方式
2. 使用EfficientNet

#### 后续 not used but worth try

1. DiVE 蒸馏提升长尾问题表现
2. Simim 训一个 swinv2 的预训练模型
3. 优化 re-distribute-label
