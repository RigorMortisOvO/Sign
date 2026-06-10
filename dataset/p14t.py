import torch
import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from spamo.constants import *
import random


class Phoenix14T(torch.utils.data.Dataset):
    """
    Phoenix14T手语数据集的数据集类。

    该类用于处理视频特征和标注的加载，支持空间特征（spatial）和时空特征（spatiotemporal）两种类型，
    为手语翻译任务提供数据访问接口。
    """

    def __init__(
            self,
            anno_root: str,
            vid_root: str,
            feat_root: str,
            mae_feat_root: str,
            mode: str = 'dev',
            spatial: bool = False,
            spatiotemporal: bool = False,
            spatial_postfix: str = '',
            spatiotemporal_postfix: Union[str, List[str]] = '',

            # context_path: str = None,  # 新增：语境对话文件的路径
            context_path: Optional[str] = None  # 新增：上下文文件路径（可选）===============================================================
    ):
        """
        初始化Phoenix14T数据集

        参数:
            anno_root: 标注文件的根目录
            vid_root: 视频文件的根目录
            feat_root: 空间特征（如ViT提取的静态特征）的根目录
            mae_feat_root: 时空特征（如VideoMAE提取的动态特征）的根目录
            mode: 数据集分割模式（'train'表示训练集，'dev'表示验证集，'test'表示测试集）
            spatial: 是否加载空间特征
            spatiotemporal: 是否加载时空特征
            spatial_postfix: 空间特征文件名的后缀（用于区分不同提取方式的特征）
            spatiotemporal_postfix: 时空特征文件名的后缀，可传入单个字符串或字符串列表（支持多组特征）
            context_path: 上下文.npy文件路径，文件中需包含fileid、s1、s2、s3字段
        """
        super().__init__()

        self.anno_root = Path(anno_root)  # 转换为Path对象，方便路径操作
        self.vid_root = Path(vid_root)
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.mode = mode  # 存储数据集分割模式
        self.spatial = spatial  # 是否使用空间特征的标志
        self.spatiotemporal = spatiotemporal  # 是否使用时空特征的标志
        self.spatial_postfix = spatial_postfix  # 空间特征文件后缀
        self.spatiotemporal_postfix = spatiotemporal_postfix  # 时空特征文件后缀

        # 新增：上下文相关属性初始化====================================================================================
        self.context_path = Path(context_path) if context_path else None  # 上下文文件路径
        # self.context_dict: Dict[str, List[str]] = {}  # 存储fileid到[s1,s2,s3]的映射  单组前文
        self.context_dict: Dict[str, List[List[str]]] = {}  # 键：fileid，值：[[s1,s2,s3], [s1,s2,s3], ...]（多组前文）

        # 输入验证：至少需要加载一种特征（空间或时空）
        if not (spatial or spatiotemporal):
            raise ValueError("必须至少启用'spatial'或'spatiotemporal'中的一种特征")

        # 加载标注数据
        anno_path = self.anno_root / f'{mode}_info_ml.npy'  # 标注文件路径（.npy格式）
        if not anno_path.exists():
            raise FileNotFoundError(f"标注文件不存在：{anno_path}")

        self.data = np.load(anno_path, allow_pickle=True).item()  # 加载标注数据（字典格式）

        # 新增：加载上下文数据（拆分为s1/s2/s3单句列表）=============================================================================
        if self.context_path:
            # self.context_dict = self._load_context_single_sentences()     # 单组前文对话
            self.context_dict = self._load_context_multi_groups()

        # 设置特征目录路径
        self.spatial_dir = self.feat_root / self.mode  # 空间特征对应分割的目录（如train/、dev/）
        self.spatiotemporal_dir = self.mae_feat_root / self.mode  # 时空特征对应分割的目录

        # 验证关键目录是否存在
        self._validate_directories()


    def _validate_directories(self) -> None:
        """验证所有必要的特征目录是否存在，确保后续特征加载不会失败"""
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"空间特征目录不存在：{self.spatial_dir}")

        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"时空特征目录不存在：{self.spatiotemporal_dir}")

        # 新增：验证上下文文件是否存在（若传入）=============================================================================
        if self.context_path and not self.context_path.exists():
            raise FileNotFoundError(f"上下文文件不存在：{self.context_path}")


    # 新增：加载上下文并拆分为s1/s2/s3单句列表================================================================================
    def _load_context_single_sentences(self) -> Dict[str, List[str]]:
        """
        加载上下文.npy文件，将每个样本的s1/s2/s3拆分为单句列表，并用文本归一化处理

        返回:
            字典，键为fileid，值为[s1_norm, s2_norm, s3_norm]（归一化后的单句列表）
        """
        # 加载上下文文件（假设格式为列表，每个元素是含fileid、s1、s2、s3的字典）
        context_data = np.load(self.context_path, allow_pickle=True).tolist()
        context_map = {}

        for item in context_data:
            # 检查上下文条目是否包含必要字段
            required_fields = ['fileid', 's1', 's2', 's3']
            if not all(field in item for field in required_fields):
                raise ValueError(f"上下文条目缺少必要字段，当前条目：{item}")

            fileid = item['fileid']
            # 拆分为单句并归一化（复用现有_normalize_text方法，确保格式统一）
            # s1_norm = self._normalize_text(item['s1'])
            # s2_norm = self._normalize_text(item['s2'])
            # s3_norm = self._normalize_text(item['s3'])

            # 不进行文本格式统一
            s1_norm = item['s1']
            s2_norm = item['s2']
            s3_norm = item['s3']
            # 存储为单句列表
            context_map[fileid] = [s1_norm, s2_norm, s3_norm]

        return context_map

    # 新增：修改上下文加载方法，支持多组前文
    def _load_context_multi_groups(self) -> Dict[str, List[List[str]]]:
        """
        加载上下文.npy文件，动态读取每个样本的所有前文组（如当前2组），每组拆分为[s1,s2,s3]单句列表

        返回:
            字典，键为fileid，值为多组前文的列表：[[s1_1,s2_1,s3_1], [s1_2,s2_2,s3_2], ...]
        """
        # 加载上下文文件（列表格式，每个元素是含fileid、s_lists的字典）
        context_data = np.load(self.context_path, allow_pickle=True).tolist()
        context_map = {}

        for item in context_data:
            # 检查上下文条目是否包含必要字段（修改：从s1/s2/s3改为s_lists）
            required_fields = ['fileid', 's_lists']
            if not all(field in item for field in required_fields):
                raise ValueError(f"p14t：上下文条目缺少必要字段（需包含fileid和s_lists），当前条目：{item}")

            fileid = item['fileid']
            multi_groups = item['s_lists']  # 多组前文：[[s1,s2,s3], [s1,s2,s3]]

            # 验证s_lists格式：必须是列表的列表（每组都是3个句子）
            if not isinstance(multi_groups, list) or len(multi_groups) == 0:
                raise ValueError(f"fileid={fileid}的s_lists必须是非空列表，当前值：{multi_groups}")

            # 处理每组前文：确保每组都是3个句子，不足则补空字符串
            processed_groups = []
            for group in multi_groups:
                if not isinstance(group, list) or len(group) < 3:
                    # 补全为3个句子（缺失部分用空字符串填充）
                    group = group[:3] + [''] * (3 - len(group))
                processed_groups.append(group)  # 每组格式：[s1, s2, s3]

            # 存储多组前文（如当前2组）
            context_map[fileid] = processed_groups

        # print(f"----p14t：成功加载上下文数据：共{len(context_map)}个fileid，每个fileid平均{np.mean([len(v) for v in context_map.values()]):.1f}组前文")
        return context_map

    def _load_spatial_features(self, file_id: str) -> torch.Tensor:
        """
        根据文件ID加载对应的空间特征

        参数:
            file_id: 样本的唯一文件标识符（用于匹配特征文件）

        返回:
            包含空间特征的张量

        异常:
            FileNotFoundError: 如果特征文件不存在时抛出
        """
        # 构建空间特征文件路径（格式：{file_id}{空间后缀}.npy）
        feat_path = self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"空间特征文件不存在：{feat_path}")

        # 加载.npy文件并转换为PyTorch张量
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal_features(self, file_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        根据文件ID加载对应的时空特征

        参数:
            file_id: 样本的唯一文件标识符

        返回:
            单个张量（单组特征）或张量列表（多组特征），包含时空特征

        异常:
            FileNotFoundError: 如果任何特征文件不存在时抛出
        """
        if isinstance(self.spatiotemporal_postfix, str):
            # 单组时空特征：构建文件路径并加载
            glor_path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not glor_path.exists():
                raise FileNotFoundError(f"时空特征文件不存在：{glor_path}")
            return torch.tensor(np.load(glor_path))
        else:
            # 多组时空特征：遍历所有后缀，分别加载并返回列表
            features = []
            for postfix in self.spatiotemporal_postfix:
                path = self.spatiotemporal_dir / f"{file_id}{postfix}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"时空特征文件不存在：{path}")
                features.append(torch.tensor(np.load(path)))
            return features

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        根据索引获取数据集中的一个样本

        参数:
            index: 要获取的样本索引

        返回:
            包含样本所有特征和元数据的字典，键包括：
            - pixel_value: 空间特征张量
            - glor_value: 时空特征张量（或列表）
            - text: 归一化后的目标文本
            - gloss: 手语对应的 gloss（手语词汇标注）
            - id: 样本唯一标识符
            - num_frames: 空间特征的帧数
            - vid_path: 原始视频路径
            - lang: 目标语言（此处为德语）
            - 其他语言文本（如en_text、es_text等，若存在）
            - original_info: 原始标注信息
        """
        data = self.data[index]  # 获取索引对应的原始标注数据
        file_id = data['fileid']  # 样本的唯一文件ID
        pixel_value = None  # 初始化空间特征
        glor_value = None  # 初始化时空特征

        # 加载空间特征（如果启用）
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                print(f"警告：{e}。返回空张量。")
                pixel_value = torch.tensor([])

        # 加载时空特征（如果启用）
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                print(f"警告：{e}。返回空张量。")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]

        # 构建结果字典，包含特征和元数据
        result = {
            'pixel_value': pixel_value,  # 空间特征
            'glor_value': glor_value,  # 时空特征
            'bool_mask_pos': None,  # 预留的掩码位置（未使用）
            'text': self._normalize_text(data['text']),  # 归一化后的目标文本
            'gloss': data['gloss'],  # 手语gloss标注
            'id': file_id,  # 样本ID
            'num_frames': len(pixel_value) if pixel_value is not None else 0,  # 空间特征的帧数
            'vid_path': str(self.vid_root / 'features' / 'fullFrame-256x256px' / data['folder']),  # 原始视频路径
            'lang': 'German',  # 目标语言（Phoenix14T的主要语言是德语）

            # 新增：上下文单句列表（若有上下文数据则返回对应列表，否则返回空列表）===================================================
            # 'context_sentences': self.context_dict.get(file_id, ['', '', ''])

            # 关键修改：返回该样本的所有前文组（如当前2组），格式：[[s1_1,s2_1,s3_1], [s1_2,s2_2,s3_2]]
            # 无上下文时返回默认1组空列表，避免模型报错
            'context_sentences': self.context_dict.get(file_id, [['', '', '']])
        }

        # 添加其他语言的文本（如英语、西班牙语、法语，若标注中存在）
        for lang in ['en', 'es', 'fr']:
            if f'{lang}_text' in data:
                result[f'{lang}_text'] = data[f'{lang}_text']

        # 存储原始标注信息供参考
        result['original_info'] = data

        return result

    def _normalize_text(self, text: str) -> str:
        """
        归一化文本：确保文本结尾以句号结束，统一文本格式

        参数:
            text: 输入文本

        返回:
            归一化后的文本
        """
        text = text.strip()  # 去除首尾空格
        if not text.endswith('.'):  # 若不以句号结尾，则添加句号
            text = f"{text}."
        return text

    def __len__(self) -> int:
        """获取数据集的样本数量"""
        return len(self.data) - 1  # 减去1可能是为了排除标注文件中的无效条目

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        """
        自定义批处理函数，用于DataLoader拼接样本

        参数:
            batch: 样本字典的列表

        返回:
            直接返回原始batch（此处未做额外处理，通常后续会在模型中处理变长特征）
        """
        return batch


"""
该代码定义了Phoenix14T类，是针对Phoenix-2014T 手语数据集的 PyTorch 数据集接口，主要作用是：
1、数据集加载与管理：通过读取标注文件（.npy格式）和预提取的视觉特征（空间特征和时空特征），将手语视频数据转换为模型可直接使用的格式，支持训练、验证、测试三种数据分割（通过mode参数控制）。
2、多特征类型支持：同时支持加载两种关键特征：
    空间特征（如通过 ViT 模型提取的静态视觉特征，反映手语的空间配置）；
    时空特征（如通过 VideoMAE 模型提取的动态特征，反映手语的运动动态）。
    特征文件通过file_id和后缀（spatial_postfix/spatiotemporal_postfix）匹配，确保正确加载对应样本的特征。
3、数据预处理与标准化：对目标文本进行归一化（确保结尾带句号），统一文本格式；同时收集样本的元数据（如视频路径、gloss 标注、多语言翻译等），为模型训练和评估提供完整信息。
4、适配 PyTorch 数据加载流程：作为torch.utils.data.Dataset的子类，实现了__getitem__和__len__方法，可直接配合DataLoader使用，支持批处理加载数据，为 SpaMo 等手语翻译模型提供输入数据。
"""