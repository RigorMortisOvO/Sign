import torch
from typing import Dict, List, Optional, Union, Tuple

import pytorch_lightning as pl  # 导入PyTorch Lightning框架，用于简化训练流程

from abc import ABC, abstractmethod  # 导入ABC和abstractmethod，用于定义抽象基类
from utils.helpers import instantiate_from_config  # 导入工具函数，用于从配置实例化对象
from torch.optim.lr_scheduler import LambdaLR  # 导入Lambda学习率调度器


class AbstractSLT(pl.LightningModule, ABC):
    """
    抽象手语翻译（Sign Language Translation, SLT）模块：
    一个抽象的PyTorch Lightning模块，定义了从视频输入到手语翻译为文本的通用接口。
    具体的视觉模型和文本模型需在子类中实现。
    """

    def __init__(
            self,
            lr: float = 0.0001,  # 学习率，默认0.0001
            monitor: Optional[str] = None,  # 监控的指标名称（如验证集BLEU分数），用于早停或 checkpoint
            scheduler_config: Optional[Dict] = None,  # 学习率调度器的配置字典
            max_length: int = 128,  # 生成文本的最大长度
            beam_size: int = 5,  # beam search的束宽，用于解码生成文本
    ):
        super().__init__()
        # 初始化模块参数
        self.lr = lr  # 存储学习率
        self.monitor = monitor  # 存储监控指标
        self.scheduler_config = scheduler_config  # 存储调度器配置
        self.max_length = max_length  # 存储生成文本的最大长度
        self.beam_size = beam_size  # 存储beam search的束宽

    @abstractmethod
    def prepare_models(self) -> None:
        """
        子类需实现此方法，用于准备视觉模型（如特征提取器）和文本模型（如翻译模型）。
        """
        pass

    @abstractmethod
    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        实现训练、验证和测试步骤的通用逻辑（如前向传播、损失计算、指标计算）。

        参数:
            inputs: 包含输入数据的字典（如视频特征、文本标签等）
            split: 当前阶段（"train"表示训练，"val"表示验证，"test"表示测试）
            batch_idx: 当前批次的索引

        返回:
            元组 (损失张量, 日志字典)，日志字典包含需要记录的指标（如损失值、BLEU分数等）
        """
        pass

    @abstractmethod
    def get_inputs(self, batch: List) -> Dict:
        """
        从数据加载器返回的原始批次数据中预处理并准备输入数据。

        参数:
            batch: 数据加载器返回的原始批次数据（通常是列表形式）

        返回:
            处理后的输入字典，包含模型所需的所有输入（如特征、标签、掩码等）
        """
        pass

    def training_step(self, batch: List, batch_idx: int) -> torch.Tensor:
        """执行训练步骤：获取输入、调用通用步骤逻辑、记录日志并返回损失。"""
        inputs = self.get_inputs(batch)  # 预处理批次数据得到模型输入
        loss, log_dict = self.shared_step(inputs, "train", batch_idx)  # 调用通用步骤计算损失和指标
        self.log_dict(log_dict, batch_size=len(inputs['text']), sync_dist=True)  # 记录训练日志（跨设备同步）
        return loss  # 返回损失用于反向传播

    def validation_step(self, batch: List, batch_idx: int) -> None:
        """执行验证步骤：获取输入、调用通用步骤逻辑并记录日志（不返回损失）。"""
        inputs = self.get_inputs(batch)  # 预处理批次数据得到模型输入
        _, log_dict = self.shared_step(inputs, "val", batch_idx)  # 调用通用步骤计算指标
        self.log_dict(log_dict, batch_size=len(inputs['text']), sync_dist=True)  # 记录验证日志（跨设备同步）

    def test_step(self, batch: List, batch_idx: int) -> None:
        """执行测试步骤：获取输入、调用通用步骤逻辑并记录日志（不返回损失）。"""
        inputs = self.get_inputs(batch)  # 预处理批次数据得到模型输入
        _, log_dict = self.shared_step(inputs, "test", batch_idx)  # 调用通用步骤计算指标
        self.log_dict(log_dict, batch_size=len(inputs['text']), sync_dist=True)  # 记录测试日志（跨设备同步）

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict]:
        """配置优化器和学习率调度器。"""
        # 初始化AdamW优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-8)

        # 如果有调度器配置，实例化LambdaLR调度器
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)  # 从配置实例化调度器逻辑
            print("设置LambdaLR学习率调度器...")
            lr_scheduler = {
                'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),  # 绑定调度器和优化器
                'interval': 'step',  # 按步骤更新学习率
                'frequency': 1  # 每1步更新一次
            }
            return [optimizer], [lr_scheduler]  # 返回优化器和调度器列表
        return optimizer  # 若无调度器，仅返回优化器


"""
该代码定义了一个名为AbstractSLT的抽象基类，是手语翻译（Sign Language Translation, SLT）任务的通用框架，基于 PyTorch Lightning 实现。其核心作用是统一手语翻译模型的训练、验证和测试流程，并为具体模型实现提供标准化接口。
主要功能：
1、统一流程抽象：
    通过继承pl.LightningModule，封装了 PyTorch Lightning 的训练（training_step）、验证（validation_step）、测试（test_step）逻辑，确保不同手语翻译模型遵循一致的训练流程。
2、标准化接口定义：
    prepare_models：要求子类实现模型初始化逻辑（如加载视觉编码器、文本解码器等）。
    get_inputs：要求子类实现数据预处理逻辑（如从原始批次中提取视频特征、文本标签等）。
    shared_step：要求子类实现核心计算逻辑（如前向传播、损失计算、翻译生成、指标评估等），该方法在训练 / 验证 / 测试阶段共享，避免代码冗余。
3、优化器与调度器配置：
    内置了configure_optimizers方法，支持 AdamW 优化器和 LambdaLR 学习率调度器，简化了优化策略的配置。
4、灵活性与扩展性：
    作为抽象类，它不依赖具体的模型结构（如视觉编码器用 ViT 还是 VideoMAE，文本解码器用 T5 还是 GPT），而是通过子类实现具体细节，适用于各种手语翻译模型的开发。
"""