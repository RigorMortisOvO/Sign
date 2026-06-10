import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.helpers import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    """
    基于配置的数据集模块，继承自PyTorch Lightning的LightningDataModule，
    用于统一管理训练、验证、测试数据集的加载和数据加载器（DataLoader）的创建。
    """

    def __init__(self, train_batch_size, val_test_batch_size, train=None, validation=None, test=None, num_workers=None):
        """
        初始化数据集模块

        参数:
            batch_size: 批处理大小，每个批次包含的样本数量
            train: 训练集的配置字典（包含数据集类和参数），为None则不创建训练集加载器
            validation: 验证集的配置字典，为None则不创建验证集加载器
            test: 测试集的配置字典，为None则不创建测试集加载器
            num_workers: 数据加载时使用的进程数，默认为batch_size的2倍
        """
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_test_batch_size
        self.test_batch_size = val_test_batch_size
        self.dataset_configs = dict()  # 存储各数据集的配置（训练/验证/测试）
        # 设置数据加载的进程数，默认使用batch_size*2以提高加载效率
        self.num_workers = num_workers if num_workers is not None else train_batch_size * 2

        # 如果提供了训练集配置，将其存入dataset_configs，并绑定训练集加载器方法
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader  # 重写父类的训练加载器方法
        # 如果提供了验证集配置，同理绑定验证集加载器方法
        if validation is not None:
            self.dataset_configs['valid'] = validation
            self.val_dataloader = self._val_dataloader  # 重写父类的验证加载器方法
        # 如果提供了测试集配置，同理绑定测试集加载器方法
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = self._test_dataloader  # 重写父类的测试加载器方法

    def setup(self, stage=None):
        """
        数据集初始化方法，在训练/验证/测试开始前执行，用于实例化各数据集

        参数:
            stage: 表示当前阶段（'fit'/'validate'/'test'），用于选择性初始化数据集
        """
        # 根据dataset_configs中的配置，实例化所有数据集（训练/验证/测试）
        # instantiate_from_config函数用于根据配置字典动态创建数据集对象
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )

    def setup(self, stage=None):
        """
        数据集初始化方法，在训练/验证/测试开始前执行，用于实例化各数据集

        参数:
            stage: 表示当前阶段（'fit'/'validate'/'test'），用于选择性初始化数据集
        """
        # 初始化数据集字典（避免重复初始化）
        self.datasets = {}

        # 1. 训练+验证阶段（fit）：需要训练集和验证集
        if stage == 'fit' or stage is None:
            # 实例化训练集（如果配置中存在）
            if 'train' in self.dataset_configs:
                self.datasets['train'] = instantiate_from_config(self.dataset_configs['train'])
            # 实例化验证集（如果配置中存在）
            if 'valid' in self.dataset_configs:
                self.datasets['valid'] = instantiate_from_config(self.dataset_configs['valid'])

        # 2. 仅验证阶段（validate）：只需要验证集
        elif stage == 'validate':
            if 'valid' in self.dataset_configs:
                self.datasets['valid'] = instantiate_from_config(self.dataset_configs['valid'])

        # 3. 测试阶段（test）：只需要测试集
        elif stage == 'test':
            if 'test' in self.dataset_configs:
                self.datasets['test'] = instantiate_from_config(self.dataset_configs['test'])

        # 其他阶段（如predict）：暂不处理，可按需扩展
        else:
            pass

    def _train_dataloader(self):
        """创建训练集的数据加载器（DataLoader）"""
        return DataLoader(
            dataset=self.datasets['train'],  # 使用训练集
            batch_size=self.train_batch_size,  # 批处理大小
            num_workers=self.num_workers,  # 加载进程数
            shuffle=True,  # 训练集需要打乱数据顺序
            collate_fn=self.datasets['train'].collate_fn  # 使用训练集自定义的数据拼接函数
        )

    def _val_dataloader(self):
        """创建验证集的数据加载器（DataLoader）"""
        return DataLoader(
            dataset=self.datasets['valid'],  # 使用验证集
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # 验证集不需要打乱顺序
            collate_fn=self.datasets['valid'].collate_fn  # 使用验证集自定义的数据拼接函数
        )

    def _test_dataloader(self):
        """创建测试集的数据加载器（DataLoader）"""
        return DataLoader(
            dataset=self.datasets['test'],  # 使用测试集
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # 测试集不需要打乱顺序
            collate_fn=self.datasets['test'].collate_fn  # 使用测试集自定义的数据拼接函数
        )


"""
该代码定义了DataModuleFromConfig类，是基于 PyTorch Lightning 框架的数据模块，主要作用是统一管理手语翻译任务中训练、验证、测试三个阶段的数据集加载流程，具体功能如下：
1、配置驱动的数据集管理：通过接收训练、验证、测试集的配置字典（包含数据集类路径和参数），动态实例化对应的数据集对象（通过instantiate_from_config函数），无需硬编码数据集类型，提高了代码的灵活性和可扩展性。
2、自动创建数据加载器：根据数据集类型（训练 / 验证 / 测试），自动生成对应的DataLoader，并根据场景设置合理参数（如训练集开启数据打乱shuffle=True，验证 / 测试集关闭）。
3、统一数据处理接口：通过绑定collate_fn（数据集自定义的数据拼接函数），确保不同阶段的样本在拼接成批次时遵循一致的处理逻辑（如处理变长序列、填充等）。
4、适配 PyTorch Lightning 工作流：作为LightningDataModule的子类，能够无缝集成到 PyTorch Lightning 的训练流程中，自动在训练、验证、测试阶段调用对应的加载器，简化了数据管理代码。
"""
