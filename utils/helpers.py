import torch
import importlib
import random
import os
import glob


def derangement(lst):
    """
    生成输入列表的错位排列（每个元素都不在原始位置上）

    参数:
        lst (list): 输入列表，长度必须至少为2

    返回:
        list: 错位排列后的列表，满足所有元素的位置与原列表完全不同

    异常:
        AssertionError: 若输入列表长度小于2则触发断言错误
    """
    # 确保输入列表至少有2个元素（否则无法生成错位排列）
    assert len(lst) > 1, "List must have at least two elements."

    # 循环打乱列表直到满足错位条件
    while True:
        # 复制原列表并打乱顺序
        shuffled = lst[:]
        random.shuffle(shuffled)
        # 检查所有元素是否都不在原来的位置上
        if all(original != shuffled[i] for i, original in enumerate(lst)):
            return shuffled


def normalize(x):
    """
    对张量进行L2归一化（按最后一个维度）

    参数:
        x (torch.Tensor): 输入张量，维度不限

    返回:
        torch.Tensor: 归一化后的张量，形状与输入相同，且最后一个维度的L2范数为1
    """
    # 计算最后一个维度的L2范数，保持维度以便广播除法
    return x / x.norm(dim=-1, keepdim=True)


def instantiate_from_config(config):
    """
    根据配置字典实例化对象（工厂模式）

    参数:
        config (dict): 配置字典，必须包含'target'键（指定类/函数的路径字符串）和可选的'params'键（构造参数）

    返回:
        object: 根据配置实例化的对象

    异常:
        KeyError: 若配置字典中没有'target'键则抛出
    """
    if 'target' not in config:
        raise KeyError('Expected key "target" to instantiate.')
    # 从配置中获取目标类/函数并传入参数实例化
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """
    从字符串引用中获取对应的对象（类或函数）

    参数:
        string (str): 对象的字符串引用，格式为"模块名.类名/函数名"（如"torch.nn.Linear"）
        reload (bool): 是否重新加载模块，默认为False（用于动态更新代码时强制刷新模块）

    返回:
        object: 字符串引用对应的类或函数
    """
    # 分割模块名和类/函数名（如"module.Class"分割为("module", "Class")）
    module, cls = string.rsplit('.', 1)
    if reload:
        # 重新加载模块（确保使用最新代码）
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 导入模块并获取目标对象
    return getattr(importlib.import_module(module, package=None), cls)


def create_mask(seq_lengths: list, device="cpu"):
    """
    根据序列长度列表创建掩码张量（标记有效序列位置）

    参数:
        seq_lengths (list): 每个序列的实际长度列表（如[5, 3, 7]表示3个序列分别长5、3、7）
        device (str): 掩码张量所在的设备（如"cpu"或"cuda"）

    返回:
        torch.Tensor: 形状为[batch_size, max_seq_len]的布尔张量，
                      其中True表示对应位置为有效序列内容，False表示填充（padding）位置
    """
    # 计算批次中最长序列的长度
    max_len = max(seq_lengths)
    # 生成掩码：对于每个序列，前seq_lengths[i]个位置为True，其余为False
    # 实现逻辑：用.arange生成0到max_len-1的序列，与每个序列长度比较（广播机制）
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.to(torch.bool)


def get_img_list(ds_name, vid_root, path):
    """
    根据数据集名称获取视频帧图像的路径列表（适配不同数据集的文件结构）

    参数:
        ds_name (str): 数据集名称（支持"Phoenix14T"和"CSL-Daily"）
        vid_root (str): 视频数据根目录
        path (str): 视频对应的相对路径（数据集内部的路径标识）

    返回:
        list: 排序后的视频帧图像路径列表（按帧顺序排列）

    异常:
        ValueError: 若数据集名称不支持则抛出
    """
    # 处理不同数据集的路径格式
    if ds_name == 'Phoenix14T':
        # Phoenix14T数据集的帧路径构造
        img_path = os.path.join(vid_root, 'features', 'fullFrame-256x256px', path)
    elif ds_name == 'CSL-Daily':
        # CSL-Daily数据集的帧路径构造
        img_path = os.path.join(vid_root, 'CSL-Daily_256x256px', path)
    else:
        raise ValueError(f"Dataset {ds_name} is not supported.")
    # 获取所有图像路径并按文件名排序（确保帧顺序正确）
    return sorted(glob.glob(img_path))


# 代码来源：https://stackoverflow.com/questions/77782599/how-can-i-extract-all-the-frames-from-a-particular-time-interval-in-a-video
def read_video(fname, start_time=None, end_time=None):
    """
    从视频文件中提取指定时间范围内的帧

    参数:
        fname (str): 视频文件路径
        start_time (float or None): 开始时间（秒），为None时从视频开头提取
        end_time (float or None): 结束时间（秒），为None时提取到视频结尾

    返回:
        list: 提取的帧（PIL.Image对象）列表，若出错则返回空列表
    """
    try:
        # 打开视频容器
        container = av.open(fname)
        # 计算视频总时长（秒）
        duration = container.duration * (1 / av.time_base)
        # 处理默认时间范围
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = duration
        # 检查时间范围有效性
        if start_time >= end_time:
            print("Start time must be less than end time")
            return []
        if end_time > duration:
            print("End time exceeds video duration")
            return []
        # 获取视频流（默认取第一个视频流）
        stream = container.streams.video[0]
        # 定位到开始时间（基于视频流的时间基准）
        container.seek(int(start_time / stream.time_base), stream=stream)
        frames = []
        # 解码并收集指定时间范围内的帧
        for frame in container.decode(stream):
            if frame.time > end_time:
                break  # 超过结束时间则停止
            elif frame.time < start_time:
                continue  # 未到开始时间则跳过
            else:
                # 将帧转换为PIL图像并添加到列表
                frames.append(frame.to_image())
        return frames
    except Exception as e:
        print(e)
        return []


def sliding_window_for_list(data_list, window_size, overlap_size):
    """
    对列表应用滑动窗口，生成重叠的子列表（用于序列分割）

    参数:
        data_list (list): 输入列表（如视频帧序列、特征序列等）
        window_size (int): 每个子列表的长度（窗口大小）
        overlap_size (int): 相邻窗口之间的重叠长度

    返回:
        list of lists: 滑动窗口处理后的子列表集合，每个子列表长度为window_size
    """
    # 计算步长（窗口每次移动的距离 = 窗口大小 - 重叠大小）
    step_size = window_size - overlap_size
    # 生成所有不越界的子列表：从索引0开始，按步长移动，直到窗口末端不超过列表长度
    windows = [data_list[i:i + window_size] for i in range(0, len(data_list), step_size) if
               i + window_size <= len(data_list)]
    return windows


"""
该文件SpaMo/utils/helpers.py是 SpaMo 手语翻译项目中的工具函数集合，提供了多种通用辅助功能，支撑模型的数据处理、特征工程、对象实例化等核心环节，具体作用如下：
1、序列与数据处理：
    derangement：生成错位排列，可用于数据增强（如打乱样本关联避免模型过拟合）或对比学习中的负样本构造。
    sliding_window_for_list：通过滑动窗口分割长序列（如视频帧、时序特征），支持重叠窗口，适用于将长视频切分为可处理的子片段。
    create_mask：生成序列掩码，在 Transformer 等模型中标记有效序列位置，忽略填充（padding）部分，确保模型聚焦于真实数据。
2、视觉数据处理：
    get_img_list：适配不同手语数据集（Phoenix14T、CSL-Daily）的文件结构，获取视频帧路径并排序，为后续特征提取提供统一接口。
    read_video：从视频中提取指定时间范围的帧，支持视频预处理阶段的帧采样，依赖av库实现高效解码。
3、特征与张量处理：
    normalize：对张量进行 L2 归一化（按最后一维），常用于视觉特征的标准化，确保不同特征的尺度一致性。
4、动态对象实例化：
    get_obj_from_str与instantiate_from_config：通过字符串引用动态加载类或函数，结合配置字典实现对象实例化（工厂模式），提高代码灵活性，支持通过配置文件定义模型组件（如不同编码器、投影层）。
"""