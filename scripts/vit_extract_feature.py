import argparse  # 用于解析命令行参数
import os  # 用于文件路径和目录操作
import os.path as osp  # 路径操作的便捷别名
import glob  # 用于文件路径匹配
import tqdm  # 用于显示进度条
import torch  # PyTorch深度学习框架
import numpy as np  # 用于数值计算
import torch.nn.functional as F  # PyTorch的函数接口
from PIL import Image  # 用于图像加载和处理
from transformers import AutoImageProcessor, CLIPVisionModel  # 用于加载CLIP视觉模型和图像处理器

import sys

sys.path.append('./')  # 将当前目录添加到系统路径，确保能导入自定义模块

from utils.s2wrapper import forward as multiscale_forward  # 导入多尺度特征提取函数
from utils.helpers import read_video, get_img_list  # 导入视频读取和图像列表获取工具函数

# 设置随机种子，保证实验结果可复现
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)  # 设置numpy随机种子
torch.manual_seed(_GLOBAL_SEED)  # 设置PyTorch随机种子


class ViTFeatureReader(object):
    """
    使用ViT（视觉Transformer）模型提取视频的空间特征的工具类
    基于CLIP的视觉模型，支持多尺度特征提取
    """

    def __init__(
            self,
            model_name='openai/clip-vit-large-patch14',  # ViT预训练模型名称（默认CLIP的ViT-L/14）
            cache_dir=None,  # 模型缓存目录
            device='cuda:0',  # 运行设备（CPU或GPU）
            s2_mode='s2wrapping',  # 多尺度处理模式（如's2wrapping'表示多尺度包装）
            scales=[1, 2],  # 多尺度处理的尺度列表
            nth_layer=-1  # 提取特征的网络层索引（-1表示最后一层）
    ):
        self.s2_mode = s2_mode  # 存储多尺度模式
        self.device = device  # 存储设备信息
        self.scales = scales  # 存储多尺度的尺度参数
        self.nth_layer = nth_layer  # 存储目标特征层索引

        # 加载CLIP视觉模型，启用隐藏层输出，移至指定设备并设置为评估模式
        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(device).eval()

        # 初始化图像处理器（用于图像预处理，如归一化、尺寸调整等）
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()  # 禁用梯度计算，节省内存并加速推理
    def forward_features(self, inputs):
        """
        前向传播获取模型指定层的特征

        参数:
            inputs: 预处理后的图像张量（批次）

        返回:
            指定层的特征张量
        """
        # 获取所有隐藏层输出，选择指定层的特征
        outputs = self.model(inputs).hidden_states
        outputs = outputs[self.nth_layer]
        return outputs

    @torch.no_grad()  # 禁用梯度计算
    def get_feats(self, video):
        """
        提取视频帧的空间特征

        参数:
            video: 视频帧列表（每个元素为PIL图像）

        返回:
            提取的特征张量（[CLS] token的特征）
        """
        # 预处理视频帧：转换为模型输入格式（张量）并移至指定设备
        inputs = self.image_processor(list(video), return_tensors="pt").to(self.device).pixel_values

        # 若启用多尺度模式，使用多尺度特征提取函数
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(self.forward_features, inputs, scales=self.scales, num_prefix_token=1)
        else:
            # 否则直接通过模型提取特征
            outputs = self.forward_features(inputs)

        # 返回[CLS] token的特征（第一个token，通常用于整体表示）
        return outputs[:, 0]


def get_parser():
    """
    定义命令行参数解析器，用于接收用户输入的配置参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='标注文件（.npy）所在目录', required=True)
    parser.add_argument('--video_root', help='视频文件所在根目录', required=True)
    parser.add_argument('--device', help='运行设备（如cuda:0或cpu）', default='cuda:0')
    parser.add_argument('--s2_mode', default='', help='多尺度处理模式（如s2wrapping）')
    parser.add_argument('--scales', nargs='+', type=int, help='多尺度处理的尺度列表', default=[])
    parser.add_argument('--batch_size', type=int, default=32, help='批量处理的视频帧数量')
    parser.add_argument('--nth_layer', type=int, default=-1, help='提取特征的网络层索引')
    parser.add_argument('--cache_dir', help='模型缓存目录', default=None)
    parser.add_argument('--save_dir', help='特征保存目录', required=True)
    parser.add_argument('--model_name', help='ViT模型名称', default='openai/clip-vit-large-patch14')
    return parser


def get_iterator(args, mode):
    """
    生成数据迭代器，用于按批次处理视频并提取特征

    参数:
        args: 命令行参数配置
        mode: 数据集模式（train/dev/test）

    返回:
        iterate: 生成器函数，用于迭代输出特征、文件ID和时间戳
        num: 数据集中的样本数量
    """
    batch_size = args.batch_size  # 批量大小

    # 加载标注数据（包含视频路径、文件ID等元信息）
    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    num = len(data) - 1  # 样本数量（减1排除可能的无效条目）
    ds_name = osp.split(args.anno_root)[-1]  # 数据集名称（从标注目录路径提取）

    # 初始化ViT特征提取器
    reader = ViTFeatureReader(
        args.model_name,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )

    def iterate():
        """生成器函数，逐样本处理视频并提取特征"""
        for i in range(num):
            fname = data[i]['folder']  # 视频文件夹路径（来自标注数据）

            # 处理Phoenix14T和CSL-Daily数据集（按图像列表加载视频）
            if ds_name == 'Phoenix14T' or ds_name == 'CSL-Daily':
                # 获取视频对应的图像文件列表（每一帧为一个图像文件）
                image_list = get_img_list(ds_name, args.video_root, fname)
                # 将图像文件转换为PIL图像列表
                videos = [Image.open(image).convert('RGB') for image in image_list]

                video_feats = []
                # 按批次提取特征（减少设备IO，提高效率）
                for j in range(0, len(videos), batch_size):
                    video_batch = videos[j:min(j + batch_size, len(videos))]  # 当前批次的视频帧
                    feats = reader.get_feats(video_batch).cpu().numpy()  # 提取特征并转移到CPU，转换为numpy数组
                    video_feats.append(feats)

                # 拼接所有批次的特征，作为该视频的完整特征
                yield np.concatenate(video_feats, axis=0), data[i]['fileid'], None

            # 处理How2Sign数据集（按时间戳从视频文件中提取片段）
            else:
                if ds_name == 'How2Sign':
                    # 获取视频片段的起始和结束时间戳（来自标注数据）
                    start_time, end_time = data[i]['original_info']['START_REALIGNED'], data[i]['original_info'][
                        'END_REALIGNED']
                    # 根据时间戳读取视频片段（返回帧列表）
                    videos = read_video(fname, start_time=start_time, end_time=end_time)

                if len(videos) > 0:  # 确保视频片段有效
                    video_feats = []
                    # 按批次提取特征
                    for j in range(0, len(videos), batch_size):
                        video_batch = videos[j:min(j + batch_size, len(videos))]
                        feats = reader.get_feats(video_batch).cpu().numpy()
                        video_feats.append(feats)
                    # 拼接特征并返回（包含起始时间戳用于文件名）
                    yield np.concatenate(video_feats, axis=0), data[i]['fileid'], str(start_time)
                else:  # 视频片段无效时返回空特征
                    yield [], data[i]['fileid'], str(start_time)

    return iterate, num


def main():
    """主函数：解析参数、初始化配置、提取并保存特征"""
    # 数据集模式（默认为训练、验证、测试集）
    mode = ["dev", "test", "train"]
    for m in mode:
        parser = get_parser()
        args = parser.parse_args()  # 解析命令行参数

        ds_name = osp.split(args.anno_root)[-1]  # 数据集名称
        _model_name = os.path.split(args.model_name)[-1]  # 模型名称（从路径提取）
        # 特征保存的根目录（按模型名称和数据集名称区分）
        fname = f'{_model_name}_feat_{ds_name}'
        # 创建保存目录（递归创建，确保目录存在）
        os.makedirs(osp.join(args.save_dir, fname, m), exist_ok=True)

        # 处理不同数据集的模式名称映射（如How2Sign的验证集名为'val'而非'dev'）
        if ds_name == 'How2Sign':
            if m == 'dev':
                _m = 'val'
            else:
                _m = m
        elif ds_name == 'NIASL2021':
            if m == 'dev': _m = 'validation'
        else:
            _m = m

        # 获取数据迭代器和样本数量
        generator, num = get_iterator(args, _m)
        iterator = generator()

        # 遍历迭代器，提取并保存特征
        for vit_feat in tqdm.tqdm(iterator, total=num):  # tqdm显示进度条
            feats, id, st = vit_feat  # 特征、文件ID、起始时间戳
            save_path = osp.join(args.save_dir, fname, m)  # 保存路径

            # 特征文件后缀（包含多尺度模式等参数信息）
            postfix = ""
            if args.s2_mode != "":
                postfix = f"_{args.s2_mode}"
            if len(args.scales) == 3:
                postfix = f'{postfix}_large'
            # 若有起始时间戳，添加到文件名后缀（用于How2Sign数据集）
            if st is not None:
                postfix = f'_{st}{postfix}'

            # 保存特征为.npy文件（文件名格式：{fileid}{后缀}.npy）
            np.save(osp.join(save_path, f'{id}{postfix}.npy'), feats)


if __name__ == "__main__":
    main()


"""
该代码是基于 ViT（视觉 Transformer）模型提取手语视频空间特征的工具脚本，是 SpaMo（手语翻译框架）的关键预处理环节，主要作用如下：
1、核心功能：空间特征提取
    使用 CLIP 预训练的 ViT 模型（如openai/clip-vit-large-patch14）提取视频帧的空间特征，捕捉手语的静态视觉信息（如手部形状、肢体位置、背景配置等空间结构）。支持从模型指定层提取特征，并通过[CLS] token 获取每帧的整体特征表示。
2、多尺度特征支持
    提供--s2_mode s2wrapping参数启用多尺度特征提取（通过multiscale_forward函数），结合不同尺度（--scales指定）的图像特征，增强对不同大小手语区域的捕捉能力，提升特征的丰富性。
3、数据集适配
    兼容三种手语数据集：
        Phoenix14T 和 CSL-Daily：从图像序列（每一帧为单独图像文件）加载视频并提取特征；
        How2Sign：根据标注的时间戳从视频文件中截取片段，再提取特征，确保特征与手语内容的时间对齐。
4、批量处理与保存
    支持批量处理视频帧（--batch_size控制）以提高效率，提取的特征以.npy格式保存，文件名包含样本唯一 ID 和参数信息（如多尺度模式、时间戳），方便后续 SpaMo 模型加载作为空间配置输入。
"""