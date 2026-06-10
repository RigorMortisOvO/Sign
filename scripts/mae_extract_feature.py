import os
import numpy as np
import torch
import argparse
import tqdm
import os.path as osp
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor

import sys

sys.path.append('./')  # 将当前目录添加到系统路径，确保能导入自定义模块

from utils.helpers import sliding_window_for_list, read_video, get_img_list

# 设置随机种子，保证结果可复现
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True  # 启用CUDA加速的基准测试模式，提高训练速度


class VideoMAEFeatureReader(object):
    """
    使用VideoMAE模型提取视频的时空特征（运动特征）的工具类
    """

    def __init__(
            self,
            model_name='MCG-NJU/videomae-large',  # VideoMAE预训练模型名称
            cache_dir=None,  # 模型缓存目录
            device='cuda:0',  # 运行设备（CPU或GPU）
            overlap_size=0,  # 滑动窗口的重叠大小
            nth_layer=-1  # 提取特征的网络层索引（-1表示最后一层）
    ):
        self.device = device  # 存储设备信息
        self.overlap_size = overlap_size  # 存储滑动窗口重叠大小
        self.nth_layer = nth_layer  # 存储目标特征层索引

        # 初始化图像处理器（用于视频帧的预处理，如归一化、尺寸调整等）
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        # 加载VideoMAE模型并移至指定设备，设置为评估模式（关闭 dropout 等训练特有层）
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()  # 禁用梯度计算，节省内存并加速推理
    def get_feats(self, video):
        """
        提取视频的特征

        参数:
            video: 视频帧列表（每个元素为PIL图像）

        返回:
            提取的视频特征张量
        """
        # 预处理视频帧：转换为模型输入格式（张量）并移至指定设备
        inputs = self.image_processor(images=video, return_tensors="pt").to(self.device)

        # 模型前向传播，获取所有隐藏层输出（hidden_states包含各层特征）
        outputs = self.model(**inputs, output_hidden_states=True).hidden_states

        # 选择指定层的特征（默认最后一层），并取cls token（[CLS]位置的特征，通常用于整体表示）
        outputs = outputs[self.nth_layer]
        outputs = outputs[:, 0]  # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]

        return outputs


def get_parser():
    """
    定义命令行参数解析器，用于接收用户输入的配置参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='标注文件（.npy）所在目录', required=True)
    parser.add_argument('--video_root', help='视频文件所在根目录', required=True)
    parser.add_argument('--save_dir', help='特征保存目录', required=True)
    parser.add_argument('--model_name', help='VideoMAE模型名称', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=32, help='批量处理的视频片段数量')
    parser.add_argument('--device', help='运行设备（如cuda:0或cpu）', default='cpu')
    parser.add_argument('--overlap_size', type=int, default=8, help='滑动窗口的重叠帧数')
    parser.add_argument('--mode', nargs='+', type=str, help='数据集模式（如train/dev/test）')
    parser.add_argument('--nth_layer', type=int, default=-1, help='提取特征的网络层索引')
    parser.add_argument('--cache_dir', help='模型缓存目录', default=None)
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

    # 初始化VideoMAE特征提取器
    reader = VideoMAEFeatureReader(
        args.model_name,
        device=args.device,
        overlap_size=args.overlap_size,
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

                # 确保视频帧数不少于16（VideoMAE模型的输入要求），不足则用最后一帧补全
                if len(image_list) < 16:
                    len_diff = 16 - len(image_list)
                    image_list.extend([image_list[-1]] * (16 - len(image_list)))
                # 用滑动窗口分割图像列表（窗口大小16帧，控制时间维度的特征捕捉）
                image_list_chunks = sliding_window_for_list(image_list, window_size=16, overlap_size=args.overlap_size)

                videos = []
                # 将每个窗口的图像转换为PIL图像列表
                for image_list in image_list_chunks:
                    videos.append([Image.open(image).convert('RGB') for image in image_list])

                video_feats = []
                # 按批次提取特征（减少设备IO，提高效率）
                for j in range(0, len(videos), batch_size):
                    video_batch = videos[j:min(j + batch_size, len(videos))]  # 当前批次的视频片段
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
                        # 补全帧数至不少于16
                        if len(videos) < 16:
                            len_diff = 16 - len(videos)
                            videos.extend([videos[-1]] * (16 - len(videos)))

                        # 滑动窗口分割视频片段
                        videos = sliding_window_for_list(videos, window_size=16, overlap_size=args.overlap_size)

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
    parser = get_parser()
    args = parser.parse_args()  # 解析命令行参数

    # 数据集模式（默认为训练、验证、测试集）
    mode = ["dev", "test", "train"]
    for m in mode:
        ds_name = osp.split(args.anno_root)[-1]  # 数据集名称
        # 特征保存的根目录（按数据集名称区分）
        fname = f'mae_feat_{ds_name}'
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
            # 特征文件后缀（包含滑动窗口重叠大小，区分不同参数的特征）
            postfix = f'_overlap-{args.overlap_size}'

            # 若有起始时间戳，添加到文件名后缀（用于How2Sign数据集）
            if st is not None:
                postfix = f'_{st}{postfix}'

            # 保存特征为.npy文件（文件名格式：{fileid}{后缀}.npy）
            np.save(osp.join(save_path, f'{id}{postfix}.npy'), feats)


if __name__ == "__main__":
    main()


"""
该代码是基于 VideoMAE 模型提取手语视频时空特征（运动特征）的工具脚本，主要用于为手语翻译模型（如 SpaMo）预处理视频数据，具体作用如下：
1、特征提取核心功能：
    使用预训练的 VideoMAE 模型（一种专门用于视频理解的 Transformer 模型）提取视频的时空特征，捕捉手语的运动动态（如手部动作、身体姿态变化等时间维度信息）。
    支持从指定网络层提取特征（默认最后一层），并通过[CLS] token 获取视频片段的整体特征表示。
2、数据集适配：
    支持三种手语数据集：Phoenix14T、CSL-Daily、How2Sign，针对不同数据集的视频存储格式（图像序列或视频文件）设计了对应的加载逻辑。
    处理 How2Sign 时，会根据标注的时间戳截取视频片段，确保特征与手语内容对齐。
3、视频预处理：
    通过滑动窗口（窗口大小 16 帧，可配置重叠大小）分割视频，平衡时间分辨率和计算效率，捕捉连续动作的动态信息。
    对帧数不足 16 的视频片段进行补全（重复最后一帧），满足模型输入要求。
4、批量处理与保存：
    支持批量提取特征，减少设备 IO 开销，提高处理速度。
    提取的特征以.npy格式保存，文件名包含样本唯一 ID 和参数信息（如重叠大小），方便后续模型加载使用。
"""