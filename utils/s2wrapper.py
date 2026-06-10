#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
from einops import rearrange  # 用于张量维度重排的工具库

import sys

sys.path.append('./')  # 将当前目录添加到系统路径，确保模块导入正常


def split_chessboard(x, num_split):
    """
    将输入图像张量按"棋盘格"方式分割为多个子正方形，并在batch维度拼接

    参数:
        x (torch.Tensor): 输入张量，形状为 [b, c, h, w]，其中b=批量大小，c=通道数，h=高度，w=宽度
        num_split (int): 分割数量，沿高度和宽度方向各分割为num_split份，总计num_split²个子块

    返回:
        torch.Tensor: 分割后拼接的张量，形状为 [b*num_split², c, h/num_split, w/num_split]
    """
    B, C, H, W = x.shape  # 获取输入张量的维度信息
    # 确保图像高度和宽度能被num_split整除，否则无法均匀分割
    assert H % num_split == 0 and W % num_split == 0
    # 计算每个子块的高度和宽度
    h, w = H // num_split, W // num_split
    # 循环遍历所有分割位置，提取子块并在batch维度拼接
    # i表示行方向分割索引，j表示列方向分割索引
    x_split = torch.cat(
        [x[:, :, i * h:(i + 1) * h, j * w:(j + 1) * w] for i in range(num_split) for j in range(num_split)], dim=0)
    return x_split


def merge_chessboard(x, num_split):
    """
    将按棋盘格分割的子块张量合并回原始完整图像张量（split_chessboard的逆操作）

    参数:
        x (torch.Tensor): 输入张量，形状为 [b, c, h, w]，其中b包含num_split²个子块的批量
        num_split (int): 分割数量，需与split_chessboard时的num_split一致

    返回:
        torch.Tensor: 合并后的完整张量，形状为 [b/(num_split²), c, h*num_split, w*num_split]
    """
    B, C, H, W = x.shape  # 获取输入张量的维度信息
    # 确保batch大小能被num_split²整除，否则无法还原为原始批量
    assert B % (num_split ** 2) == 0
    # 计算原始图像的批量大小（每个完整图像对应num_split²个子块）
    b = B // (num_split ** 2)
    # 先按列拼接子块（j循环），再按行拼接（i循环），还原完整图像
    x_merge = torch.cat(
        [torch.cat([x[(i * num_split + j) * b:(i * num_split + j + 1) * b] for j in range(num_split)], dim=-1)
         for i in range(num_split)], dim=-2)
    return x_merge


def forward(
        model,
        input,
        scales=None,
        img_sizes=None,
        max_split_size=None,
        resize_output_to_idx=0,
        num_prefix_token=0,
        output_shape='bnc',
):
    """
    多尺度图像特征提取的前向传播函数，支持大图像分割处理、多尺度特征融合

    参数:
        model: 用于特征提取的模型（如ViT或卷积网络）
        input (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]，需为正方形图像
        scales (list[float], optional): 多尺度缩放比例，与img_sizes二选一
        img_sizes (list[int], optional): 多尺度图像尺寸，与scales二选一
        max_split_size (int, optional): 图像分割的最大子块尺寸，默认与输入图像尺寸相同
        resize_output_to_idx (int, optional): 选择哪个尺度的输出尺寸作为基准，其他尺度向其对齐
        num_prefix_token (int, optional): 模型输出中前缀token的数量（如ViT的cls token）
        output_shape (str, optional): 输出特征的形状模式，'bnc'表示[B, N, C]（如ViT的序列特征），
                                     'bchw'表示[B, C, H, W]（如卷积网络的空间特征）

    返回:
        torch.Tensor: 融合后的多尺度特征张量，形状由output_shape决定
    """
    # 输入合法性检查
    assert input.dim() == 4, "输入图像必须为4维张量，形状为BxCxHxW。"
    assert input.shape[2] == input.shape[3], "目前仅支持正方形图像。"
    assert output_shape in ['bnc', 'bchw'], "输出形状模式必须为'bnc'（如ViT）或'bchw'（如卷积网络）。"
    assert output_shape == 'bnc' or num_prefix_token == 0, "卷积网络输出模式下不应存在前缀token。"

    b, c, input_size, _ = input.shape  # 获取输入图像的批量大小、通道数和尺寸

    # 确定各尺度的图像尺寸
    assert scales is not None or img_sizes is not None, "必须指定scales（缩放比例）或img_sizes（图像尺寸）。"
    # 若未提供img_sizes，则根据输入尺寸和scales计算各尺度尺寸
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

    # 准备多尺度输入
    max_split_size = max_split_size or input_size  # 子块最大尺寸，默认与输入尺寸相同
    # 计算每个尺度下需要分割的数量（沿高度/宽度方向）
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]
    input_multiscale = []  # 存储各尺度的分割后输入
    for size, num_split in zip(img_sizes, num_splits):
        # 将输入图像缩放到当前尺度
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        # 按棋盘格分割为子块，便于模型处理大图像
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # 对每个尺度的输入执行模型前向传播
    outs_multiscale = [model(x) for x in input_multiscale]
    # 处理前缀token（如ViT的cls token）：分离前缀token和空间特征
    if num_prefix_token > 0:
        outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]  # 提取前缀token
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]  # 保留空间特征部分
    # 若输出模式为'bnc'（如ViT的序列特征），先转换为空间特征形状[b, c, h, w]以便后续处理
    if output_shape == 'bnc':
        # 将序列特征[b, n, c]重排为空间特征[b, c, h, w]（假设h=w=√n）
        outs_multiscale = [
            rearrange(out, 'b (h w) c -> b c h w', h=int(out.shape[1] ** 0.5), w=int(out.shape[1] ** 0.5))
            for out in outs_multiscale]

    # 对每个尺度的输出，将分割的子块合并回完整特征图
    outs_multiscale = [merge_chessboard(out, num_split=num_split) for num_split, out in
                       zip(num_splits, outs_multiscale)]

    # 将所有尺度的特征图插值到同一尺寸（以resize_output_to_idx指定的尺度为基准），并在通道维度拼接
    output_size = outs_multiscale[resize_output_to_idx].shape[-2]  # 基准输出尺寸（高度/宽度）
    out = torch.cat([
        # 对每个尺度的特征图进行插值，统一尺寸
        F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size, mode='area').to(outs_multiscale[i].dtype)
        for i in range(len(outs_multiscale))
    ], dim=1)  # 在通道维度拼接多尺度特征

    # 若输出模式为'bnc'，将空间特征重排回序列特征形状
    if output_shape == 'bnc':
        out = rearrange(out, 'b c h w -> b (h w) c')
    # 若存在前缀token，将各尺度的前缀token合并后拼接到特征前
    if num_prefix_token > 0:
        # 对每个尺度的前缀token，按原始批量分割后取平均（消除分割带来的冗余）
        outs_prefix_multiscale = [torch.stack(out.split(b, dim=0), dim=0).mean(dim=0) for out in outs_prefix_multiscale]
        # 在通道维度拼接各尺度的前缀token
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        # 将前缀token拼接到特征前
        out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out


"""
该文件实现了一个多尺度图像特征提取的工具，核心功能是对输入图像进行多尺度处理、大图像分割 - 合并，并融合不同尺度的特征，主要用于增强视觉模型对不同尺度信息的捕捉能力。结合 SpaMo 项目背景（手语翻译模型，依赖空间和运动特征），其具体作用如下：
1、棋盘格分割与合并：
    split_chessboard：将大图像均匀分割为多个子块，解决模型无法直接处理超大尺寸图像的问题（如显存限制）。
    merge_chessboard：将分割的子块特征重新合并为完整特征图，恢复空间结构，是分割操作的逆过程。
2、多尺度特征提取与融合：
    forward函数是核心，支持输入图像按不同尺度（通过scales或img_sizes指定）缩放，每个尺度的图像可分割为子块输入模型。
    对各尺度的模型输出进行合并、尺寸统一（插值到同一尺度）和通道拼接，融合多尺度信息，增强特征的丰富性。
3、适配不同类型模型：
    支持两种输出模式（'bnc'适用于 ViT 等生成序列特征的模型，'bchw'适用于卷积网络等生成空间特征的模型）。
    处理模型输出中的前缀 token（如 ViT 的 cls token），确保多尺度融合时前缀信息的有效整合。
"""