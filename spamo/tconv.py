import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """
    时间卷积模块，用于对时序特征进行卷积和池化操作，提取时间维度上的特征
    """

    def __init__(self, input_size, hidden_size, conv_type=2, num_classes=-1):
        """
        初始化时间卷积模块

        参数:
            input_size: 输入特征的维度
            hidden_size: 卷积层输出特征的维度
            conv_type: 卷积类型，决定了卷积核和池化层的组合方式
            num_classes: 分类数量，若为-1则不添加最终的全连接层
        """
        super(TemporalConv, self).__init__()
        self.input_size = input_size  # 输入特征维度
        self.hidden_size = hidden_size  # 卷积层输出特征维度
        self.num_classes = num_classes  # 分类数量
        self.conv_type = conv_type  # 卷积类型

        # 根据卷积类型定义卷积核(K)和池化层(P)的序列，K后数字表示卷积核大小，P后数字表示池化核大小
        if self.conv_type == 0:
            self.kernel_size = ['K3']  # 单个3x1卷积
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]  # 5x1卷积 + 2x1池化
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]  # 5x1卷积 + 2x1池化 + 5x1卷积 + 2x1池化
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]  # 5x1卷积 + 5x1卷积 + 2x1池化
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']  # 两个5x1卷积
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']  # 5x1卷积 + 2x1池化 + 5x1卷积
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']  # 2x1池化 + 5x1卷积 + 5x1卷积
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']  # 2x1池化 + 5x1卷积 + 2x1池化 + 5x1卷积
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']  # 两个2x1池化 + 两个5x1卷积

        modules = []  # 存储卷积层和池化层的列表
        for layer_idx, ks in enumerate(self.kernel_size):
            # 确定当前层的输入特征维度：第一层或特定卷积类型的特定层使用input_size，其余使用hidden_size
            input_sz = self.input_size if layer_idx == 0 or (self.conv_type == 6 and layer_idx == 1) or (
                        self.conv_type == 7 and layer_idx == 1) or (
                                                      self.conv_type == 8 and layer_idx == 2) else self.hidden_size

            if ks[0] == 'P':
                # 池化层：添加最大池化操作，核大小为ks[1]
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                # 卷积层：添加卷积、批归一化和ReLU激活函数
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))  # 批归一化，加速训练
                modules.append(nn.ReLU(inplace=True))  # ReLU激活函数，增加非线性

        self.temporal_conv = nn.Sequential(*modules)  # 将所有层组合成序列模型

        # 如果需要分类，添加全连接层
        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        """
        根据卷积和池化操作更新特征序列的长度

        参数:
            lgt: 原始特征序列的长度

        返回:
            更新后的特征序列长度
        """
        feat_len = copy.deepcopy(lgt)  # 复制原始长度
        for ks in self.kernel_size:
            if ks[0] == 'P':
                # 池化操作：长度除以池化核大小
                feat_len = torch.div(feat_len, 2)
            else:
                # 卷积操作：长度减去(卷积核大小-1)
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        """
        前向传播过程

        参数:
            frame_feat: 输入的帧特征，形状为[batch, input_size, seq_len]
            lgt: 输入特征序列的长度

        返回:
            包含处理后的视觉特征、分类logits（若有）和更新后长度的字典
        """
        visual_feat = self.temporal_conv(frame_feat)  # 通过时间卷积模块处理特征
        lgt = self.update_lgt(lgt)  # 更新特征序列长度
        # 若需要分类，通过全连接层计算logits并调整维度；否则为None
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),  # 调整维度为[seq_len, batch, hidden_size]
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,  # 调整logits维度
            "feat_len": lgt.cpu(),  # 更新后的特征长度（转移到CPU）
        }


class ResidualBlock(nn.Module):
    """
    残差块，用于增强特征传播，缓解深层网络的梯度消失问题
    """

    def __init__(self, channels, kernel_size=3, padding=1):
        """
        初始化残差块

        参数:
            channels: 输入和输出特征的通道数（保持一致）
            kernel_size: 卷积核大小
            padding: 填充大小，保持卷积前后特征长度不变
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, stride=1)  # 1D卷积
        self.bn1 = nn.BatchNorm1d(channels)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, x):
        """
        前向传播：卷积+批归一化+激活+残差连接

        参数:
            x: 输入特征，形状为[batch, channels, seq_len]

        返回:
            处理后的特征，形状与输入一致
        """
        residual = x  # 保存输入作为残差
        out = self.conv1(x)  # 卷积操作
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # 激活
        out = out + residual  # 残差连接（元素级相加）
        out = self.relu(out)  # 再次激活
        return out


class GlorTemporalConv(nn.Module):
    """
    基于残差块的时间卷积模块，使用扩张卷积捕捉更大范围的时间依赖
    """

    def __init__(self, input_channels, output_channels, dilation_rate=1):
        """
        初始化GlorTemporalConv模块

        参数:
            input_channels: 输入特征的通道数
            output_channels: 输出特征的通道数
            dilation_rate: 扩张率，控制卷积的感受野
        """
        super().__init__()

        self.layers = nn.ModuleList()  # 存储网络层的列表
        # 添加扩张卷积层：通过dilation_rate扩大感受野，无需增加卷积核大小
        self.layers.append(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding=dilation_rate,
                      dilation=dilation_rate)
        )
        # 添加残差块，增强特征学习能力
        self.layers.append(ResidualBlock(output_channels))

    def forward(self, x):
        """
        前向传播过程

        参数:
            x: 输入特征，形状为[batch, seq_len, input_channels]

        返回:
            处理后的特征，形状为[batch, seq_len, output_channels]
        """
        x = x.permute(0, 2, 1)  # 调整维度为[batch, input_channels, seq_len]以适应1D卷积
        for layer in self.layers:
            x = layer(x)  # 依次通过各层
        return x.permute(0, 2, 1)  # 调整回[batch, seq_len, output_channels]

"""
该文件定义了三个用于处理时序特征的卷积模块，主要用于提取序列数据（如视频帧特征）的时间动态信息，在手语翻译等需要捕捉时间维度变化的任务中发挥作用：
1、TemporalConv：
    通过可配置的卷积核和池化层组合，对输入的时序特征进行多层处理。支持 8 种不同的卷积 - 池化组合方式（通过conv_type控制），能灵活调整特征提取的感受野和输出序列长度，适用于捕捉不同尺度的时间模式。
2、ResidualBlock：
    残差块结构，通过卷积层与输入的残差连接缓解深层网络的梯度消失问题，增强特征的传播和复用能力，提升模型对复杂特征的学习效果。
3、GlorTemporalConv：
    结合扩张卷积和残差块的时间卷积模块。扩张卷积通过设置dilation_rate在不增加计算量的情况下扩大感受野，能捕捉更长范围的时间依赖关系，适用于需要长时序上下文的场景。
"""