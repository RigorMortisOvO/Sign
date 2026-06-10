import torch
import torch.nn as nn
import re
import math
import torch.nn.functional as F
from torch.nn.attention import SDPBackend


# 代码来源参考: https://github1s.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py
class IdentityMap(nn.Module):
    """恒等映射模块，直接返回输入数据，不做任何变换"""

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        """前向传播：直接返回输入x"""
        return x

    @property
    def config(self):
        """返回模块配置信息"""
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    """简单残差块，包含层归一化和两层线性变换，用于特征增强"""

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)  # 层归一化，稳定训练

        # 两层线性变换组成的投影网络，带GELU激活函数
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        """前向传播：先归一化，再通过投影网络，最后加残差连接"""
        x = self.pre_norm(x)
        return x + self.proj(x)  # 残差连接，缓解梯度消失问题


def build_vision_projector(mm_projector_type='linear', mm_hidden_size=512, hidden_size=768, mlp_depth=1):
    """
    构建视觉特征投影器，将视觉特征映射到与语言模型兼容的维度

    参数:
        mm_projector_type: 投影器类型，支持'linear'、'mlpNx_gelu'（N为层数）、'identity'
        mm_hidden_size: 输入视觉特征的维度
        hidden_size: 输出特征的维度（需与语言模型维度匹配）
        mlp_depth: MLP投影器的默认深度（当类型为mlpNx_gelu且N未指定时使用）

    返回:
        构建好的投影器模块
    """
    # 线性投影器：单层线性变换
    if mm_projector_type == 'linear':
        return nn.Linear(mm_hidden_size, hidden_size)

    # MLP投影器：多层线性变换，带GELU激活（如mlp2x_gelu表示2层MLP）
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_projector_type)
    if mlp_gelu_match:
        # 从投影器类型中提取MLP层数，若无法提取则使用默认mlp_depth
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match.group(1).isdigit() else mlp_depth
        modules = [nn.Linear(mm_hidden_size, hidden_size)]  # 第一层：从输入维度映射到目标维度
        # 添加剩余层：每层由GELU激活和线性变换组成
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    # 恒等投影器：不改变输入特征
    if mm_projector_type == 'identity':
        return IdentityMap()

    # 不支持的投影器类型
    raise ValueError(f'未知的投影器类型: {mm_projector_type}')


# 代码来源参考: https://github1s.com/facebookresearch/jepa/blob/main/src/models/utils/modules.py
class CrossAttention(nn.Module):
    """
    交叉注意力模块，用于处理查询（Q）和键值对（K、V）来自不同模态的注意力计算
    """

    def __init__(
            self,
            dim,  # 特征维度
            num_heads=12,  # 注意力头数
            qkv_bias=False,  # 是否在Q、K、V的线性层中使用偏置
            use_sdpa=True  # 是否使用PyTorch的Scaled Dot Product Attention优化
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个注意力头的维度
        self.scale = head_dim ** -0.5  # 注意力缩放因子（1/sqrt(head_dim)）
        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # Q的线性投影层
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)  # K和V的联合线性投影层（输出维度为2*dim）
        self.proj = nn.Linear(dim, dim)  # 注意力输出的线性投影层
        self.use_sdpa = use_sdpa  # 是否使用优化的SDPA实现

    def forward(self, q, x):
        """
        前向传播：计算交叉注意力

        参数:
            q: 查询（Query）特征，形状为 (B, n, C)，其中B为批次大小，n为查询序列长度，C为特征维度
            x: 键值对（Key和Value）的源特征，形状为 (B, N, C)，其中N为键值序列长度

        返回:
            注意力计算后的查询特征，形状为 (B, n, C)
        """
        B, n, C = q.shape
        # 计算Q：线性投影后拆分为多个注意力头
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                 3)  # 形状: (B, num_heads, n, head_dim)

        B, N, C = x.shape
        # 计算K和V：线性投影后拆分为多个注意力头
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                      4)  # 形状: (2, B, num_heads, N, head_dim)
        k, v = kv[0], kv[1]  # k: (B, num_heads, N, head_dim); v: (B, num_heads, N, head_dim)

        # # 计算缩放点积注意力
        if self.use_sdpa:
            # 使用PyTorch优化的SDPA实现（更高效）
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        # 新代码----5090
        # if self.use_sdpa:
        #     # 使用PyTorch优化的SDPA实现（更高效）
        #     with torch.nn.attention.sdpa_kernel(
        #             backends=[
        #                 SDPBackend.FLASH_ATTENTION,  # 优先启用FlashAttention（最快）
        #                 SDPBackend.MATH,  # fallback到PyTorch原生实现
        #             ]
        #     ):
        #         q = F.scaled_dot_product_attention(q, k, v)
        else:
            # 手动计算注意力：Q*K^T / sqrt(head_dim) -> softmax -> 与V相乘
            xattn = (q @ k.transpose(-2, -1)) * self.scale  # 注意力分数: (B, num_heads, n, N)
            xattn = xattn.softmax(dim=-1)  # 注意力权重归一化
            q = (xattn @ v)  # 加权求和: (B, num_heads, n, head_dim)

        # 合并注意力头并通过线性投影输出
        q = q.transpose(1, 2).reshape(B, n, C)  # 形状: (B, n, C)
        q = self.proj(q)  # 最终投影

        return q


class MLP(nn.Module):
    """多层感知器，用于特征的非线性变换"""

    def __init__(
            self,
            in_features,  # 输入特征维度
            hidden_features=None,  # 隐藏层维度，默认与输入维度相同
            out_features=None,  # 输出特征维度，默认与输入维度相同
            act_layer=nn.GELU,  # 激活函数类型
            drop=0.  # Dropout概率
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层线性变换
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层线性变换
        self.drop = nn.Dropout(drop)  # Dropout层，防止过拟合

    def forward(self, x):
        """前向传播：线性变换 -> 激活 -> Dropout -> 线性变换 -> Dropout"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块，结合交叉注意力和MLP，用于跨模态特征融合
    包含层归一化和残差连接，增强特征表达能力
    """

    def __init__(
            self,
            dim,  # 特征维度
            num_heads,  # 注意力头数
            mlp_ratio=4.,  # MLP隐藏层维度相对于输入维度的比例
            qkv_bias=False,  # 交叉注意力中Q、K、V的线性层是否使用偏置
            act_layer=nn.GELU,  # 激活函数类型
            norm_layer=nn.LayerNorm  # 归一化层类型
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)  # 对键值对特征的归一化
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)  # 交叉注意力模块
        self.norm2 = norm_layer(dim)  # 对注意力输出的归一化
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP隐藏层维度
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)  # MLP模块

    def forward(self, q, x):
        """
        前向传播：交叉注意力 + 残差连接 -> MLP + 残差连接

        参数:
            q: 查询（Query）特征，形状为 (B, n, C)
            x: 键值对（Key和Value）的源特征，形状为 (B, N, C)

        返回:
            融合后的查询特征，形状为 (B, n, C)
        """
        # 交叉注意力计算：先对x归一化，再与q计算注意力，最后加残差
        y = self.xattn(q, self.norm1(x))
        q = q + y
        # MLP处理：先对q归一化，再通过MLP，最后加残差
        q = q + self.mlp(self.norm2(q))
        return q


"""
该代码实现了一系列多模态特征融合的核心组件，主要用于将视觉特征（如空间特征、运动特征）与语言模型的特征空间进行对齐和融合，是 SpaMo（手语翻译模型）中连接视觉编码器和语言模型（LLM）的关键模块。具体作用如下：
1、视觉投影器（Vision Projector）
    通过build_vision_projector函数构建不同类型的投影器，将视觉特征（如 ViT 提取的空间特征、VideoMAE 提取的运动特征）从原始维度映射到与语言模型兼容的维度。支持三种投影方式：
        线性投影（linear）：适用于简单的维度转换。
        MLP 投影（mlpNx_gelu）：通过多层感知器实现非线性映射，增强特征表达能力。
        恒等映射（identity）：不改变输入特征，适用于已对齐的特征。
2、交叉注意力机制（Cross Attention）
    实现了跨模态注意力计算（CrossAttention类），允许模型关注视觉特征中与语言任务相关的部分（如手语视频中与语义相关的动作区域）。支持使用 PyTorch 优化的scaled_dot_product_attention提升计算效率。
3、特征增强模块
    SimpleResBlock：通过残差连接和层归一化增强特征的非线性表达能力，避免深层网络的梯度消失问题。
    MLP：基础多层感知器，用于特征的非线性变换。
    CrossAttentionBlock：组合交叉注意力和 MLP，形成完整的跨模态特征融合块，通过残差连接和层归一化稳定训练，提升融合效果。
"""