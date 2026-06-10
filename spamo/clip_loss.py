# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
import torch


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    计算对比损失（Contrastive Loss），用于衡量正负样本对的相似度差异。
    适用于单模态或跨模态的对比学习任务，通过交叉熵损失迫使正样本对的相似度高于负样本对。

    参数:
        logits: 相似度矩阵，形状为 [N, N]（N为样本数量），其中logits[i][j]表示第i个样本与第j个样本的相似度得分

    返回:
        交叉熵损失值，反映正样本对（对角线元素）与负样本对（非对角线元素）的区分度
    """
    # 交叉熵损失：目标是让每个样本i与样本i（正样本）的相似度最高
    # 标签为0到N-1的整数序列（对角线索引），表示每个样本对应的正样本位置
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    实现CLIP（Contrastive Language-Image Pretraining）模型中的对比损失，用于跨模态（如图像-文本）对齐。
    通过同时计算文本到图像和图像到文本的对比损失并取平均，确保两种模态的特征空间能够相互映射。

    参数:
        similarity: 跨模态相似度矩阵，形状为 [N, N]，其中similarity[i][j]表示第i个文本与第j个图像的相似度

    返回:
        平均损失值，综合了文本到图像和图像到文本两个方向的对比损失
    """
    # 计算文本到图像的对比损失：每个文本应与对应的图像（对角线）最相似
    caption_loss = contrastive_loss(similarity)
    # 计算图像到文本的对比损失：通过转置矩阵，每个图像应与对应的文本（对角线）最相似
    image_loss = contrastive_loss(similarity.t())
    # 返回两个方向损失的平均值，平衡文本和图像模态的对齐
    return (caption_loss + image_loss) / 2.0


"""
该代码实现了 CLIP 模型中核心的跨模态对比损失函数，主要用于训练过程中对齐不同模态的特征（如手语视频的视觉特征与文本特征），具体作用如下：
1、对比损失（contrastive_loss）：
    针对单模态或跨模态的相似度矩阵，通过交叉熵损失迫使正样本对（如 “手语视频 - 对应文本”）的相似度高于所有负样本对（如 “手语视频 - 其他文本”）。它通过将对角线元素（正样本）设为目标标签，让模型学习区分正负样本。
2、CLIP 损失（clip_loss）：
    扩展了对比损失，专门用于跨模态场景（如视觉 - 文本）。它同时计算两个方向的损失：
        文本到视觉的损失（每个文本与对应视觉特征匹配）
        视觉到文本的损失（每个视觉特征与对应文本匹配）
            最终取两者的平均值，确保两种模态的特征在共享空间中能够双向对齐，提升模型对 “视觉内容 - 文本描述” 对应关系的理解能力。
"""