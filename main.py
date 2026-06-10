import argparse
import datetime
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from utils.helpers import instantiate_from_config  # 从配置实例化对象的工具函数
from spamo.callbacks import SetupCallback  # 自定义的设置回调类


def str2bool(v: Any) -> bool:
    """将字符串表示转换为布尔值

    Args:
        v: 待转换的输入值

    Returns:
        输入值对应的布尔值

    Raises:
        ArgumentTypeError: 若输入无法被解析为布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('期望布尔值')


def get_parser() -> argparse.ArgumentParser:
    """创建包含所有必要命令行选项的参数解析器

    Returns:
        配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description='SpaMo模型的训练和评估')
    parser.add_argument(
        '-c', '--config', nargs='*', metavar='base_config.yaml', default=list(),
        help='要加载的配置文件'
    )
    parser.add_argument(
        '-t', '--train', type=str2bool, default=True, nargs='?',
        help='是否以训练模式运行'
    )
    parser.add_argument(
        '--test', type=bool, default=False,
        help='是否以测试模式运行'
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help='随机数生成器的种子'
    )
    parser.add_argument(
        '-f', '--fast_dev_run', action='store_true', default=False,
        help='运行测试批次以调试'
    )
    parser.add_argument(
        '-n', '--name', type=str, const=True, default='', nargs='?',
        help='日志目录的后缀'
    )
    parser.add_argument(
        '--postfix', type=str, default='',
        help='日志目录的额外后缀'
    )
    parser.add_argument(
        '-l', '--logdir', type=str, default='logs',
        help='日志的基础目录'
    )
    parser.add_argument(
        '-r', '--resume', default=None,
        help='从检查点目录恢复训练'
    )
    parser.add_argument(
        '--no_test', type=bool, default=True,
        help='训练后是否跳过测试阶段'
    )
    parser.add_argument(
        '--ckpt', type=str, default=None,
        help='用于恢复或测试的检查点文件'
    )
    parser.add_argument(
        '-e', '--evaluation', type=str, default='mse',
        help='使用的评估指标'
    )
    return parser


def load_configs(config_paths: List[str]) -> OmegaConf:
    """加载并合并多个配置文件

    Args:
        config_paths: 配置文件路径列表

    Returns:
        合并后的配置
    """
    configs = [OmegaConf.load(cfg) for cfg in config_paths]
    return OmegaConf.merge(*configs)


def setup_logging_dirs(opt: argparse.Namespace) -> tuple:
    """设置日志目录并确定检查点路径

    Args:
        opt: 命令行参数

    Returns:
        元组 (日志目录, 检查点路径, 当前运行名称)

    Raises:
        ValueError: 若恢复的目录不存在
    """
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")  # 当前时间戳，用于命名

    if opt.resume:
        # 恢复训练：使用已存在的日志目录
        if not os.path.exists(opt.resume):
            raise ValueError(f"找不到检查点目录: {opt.resume}")

        logdir = opt.resume.rstrip("/")
        # 确定检查点文件路径（若指定）
        ckpt = os.path.join(logdir, "checkpoints", opt.ckpt) if opt.ckpt else None
        nowname = logdir.split("/")[-1]  # 运行名称为已有目录的最后一部分
    else:
        # 新训练：生成新的日志目录名称
        if opt.name:
            name = "_" + opt.name
        elif opt.config:
            # 从配置文件名提取名称
            cfg_fname = os.path.split(opt.config[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix  # 组合时间戳、名称和后缀作为运行名称
        logdir = os.path.join(opt.logdir, nowname)  # 日志目录路径
        ckpt = opt.ckpt  # 直接使用指定的检查点（若有）

    return logdir, ckpt, nowname


def configure_callbacks(
        opt: argparse.Namespace,
        model: pl.LightningModule,
        ckptdir: str,
        lightning_config: OmegaConf,
        logdir: str,
        now: str,
        config: OmegaConf
) -> List:
    """配置训练回调函数

    Args:
        opt: 命令行参数
        model: Lightning模块
        ckptdir: 检查点目录
        lightning_config: Lightning配置
        logdir: 日志目录
        now: 当前时间戳
        config: 完整配置

    Returns:
        回调函数列表
    """
    # 实例化配置中定义的回调
    callbacks = [
        instantiate_from_config(lightning_config.callback[callback])
        for callback in lightning_config.callback.keys()
    ]

    # 根据评估指标类型添加检查点和早停回调
    if opt.evaluation == "bleu":
        # BLEU指标：值越高越好
        callbacks.append(ModelCheckpoint(
            dirpath=ckptdir,
            filename="epoch={epoch:05}-step={step:07}-bleu4={val/bleu4:.2f}",
            monitor=model.monitor,  # 监控模型中定义的指标
            auto_insert_metric_name=False,
            save_top_k=1,  # 只保存最好的1个检查点
            mode="max"  # 最大化监控指标
        ))
        callbacks.append(EarlyStopping(
            monitor=model.monitor, verbose=True, patience=200, mode="max"  # 100轮无提升则早停。
        ))
    else:
        # 其他指标（如损失）：值越低越好
        callbacks.append(ModelCheckpoint(
            dirpath=ckptdir,
            filename="epoch={epoch:05}-step={step:07}-loss={val/contra_loss:.4f}",
            monitor=model.monitor,
            auto_insert_metric_name=False,
            save_top_k=1,
            mode="min"  # 最小化监控指标
        ))
        callbacks.append(EarlyStopping(
            monitor=model.monitor, verbose=True, patience=200, mode="min"
        ))

    # 添加配置日志回调（保存配置文件等）
    callbacks.append(SetupCallback(
        resume=opt.resume,
        now=now,
        logdir=logdir,
        ckptdir=ckptdir,
        cfgdir=os.path.join(logdir, "configs"),
        config=config,
        lightning_config=lightning_config
    ))

    return callbacks


def configure_logger(logger_type: str, logdir: str, nowname: str) -> Dict:
    """配置日志工具

    Args:
        logger_type: 日志工具类型
        logdir: 日志目录
        nowname: 当前运行名称

    Returns:
        日志配置字典
    """
    logger_configs = {
        "wandb": {  # Weights & Biases日志
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "id": nowname,
            }
        },
        "testtube": {  # TestTube日志
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
        "tensorboard": {  # TensorBoard日志
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "version": nowname,
                "save_dir": logdir
            }
        }
    }

    # 若指定类型不存在，默认使用tensorboard
    if logger_type not in logger_configs:
        logger_type = "tensorboard"

    return logger_configs[logger_type]


def main():
    """训练和测试的主入口函数"""
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())  # 将当前目录添加到系统路径，确保模块可导入

    # 解析命令行参数
    parser = get_parser()
    opt, _ = parser.parse_known_args()

    # 验证参数：name和resume不能同时指定
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name 和 -r/--resume 不能同时指定。"
            "如果想在新的日志文件夹中恢复训练，"
            "请结合使用 -n/--name 和 --resume_from_checkpoint"
        )

    # 设置目录和检查点路径
    logdir, ckpt, nowname = setup_logging_dirs(opt)
    ckptdir = os.path.join(logdir, "checkpoints")  # 检查点存储目录
    cfgdir = os.path.join(logdir, "configs")  # 配置文件存储目录

    # 设置随机种子以保证可复现性
    seed_everything(opt.seed)

    # 加载配置文件：恢复训练或测试时，优先加载已有配置
    if opt.resume or opt.test:
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.config = base_configs + opt.config  # 合并基础配置和新配置

    config = load_configs(opt.config)  # 合并所有配置文件
    lightning_config = config.pop("lightning", OmegaConf.create())  # 提取lightning相关配置

    # 配置训练器
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    if opt.fast_dev_run:
        trainer_config["fast_dev_run"] = True  # 快速开发模式（测试用）
    trainer_opt = argparse.Namespace(**trainer_config)  # 转换为命名空间
    lightning_config.trainer = trainer_config

    # 实例化数据模块
    data = instantiate_from_config(config.data)
    data.setup()  # 准备数据（划分训练/验证/测试集等）

    # 实例化模型
    model = instantiate_from_config(config.model)

    # 为非开发模式配置训练器的回调和日志
    if not opt.fast_dev_run:
        logger_cfg = configure_logger("wandb", logdir, nowname)  # 配置日志
        trainer_opt.logger = instantiate_from_config(logger_cfg)  # 实例化日志器

        # 配置回调函数
        trainer_opt.callbacks = configure_callbacks(
            opt, model, ckptdir, lightning_config, logdir, now, config
        )

    # 创建训练器
    trainer = Trainer(**vars(trainer_opt))

    # 运行训练或测试
    if opt.train:
        if opt.resume is not None:
            # 从检查点恢复训练
            trainer.fit(model, data, ckpt_path=ckpt)
        else:
            if ckpt is not None:
                # 加载预训练权重后开始训练
                model.load_pretrained_weights(ckpt)
                trainer.fit(model, data)
            else:
                # 从头开始训练
                trainer.fit(model, data)

            # 训练后执行测试（若不跳过）
            if not opt.no_test:
                trainer.test(model, data)
    elif opt.test:
        # 仅执行测试
        trainer.test(model, data, ckpt_path=ckpt)


if __name__ == '__main__':
    main()


"""
该文件是 SpaMo（基于空间配置和运动动态的手语翻译模型）的主程序入口，负责协调模型的训练、评估流程，处理命令行参数、配置管理、日志记录等核心功能。具体作用如下：
1、命令行参数解析：
    通过get_parser函数定义并解析所有训练 / 测试相关参数（如配置文件路径、训练模式、日志目录、检查点路径等），支持灵活控制运行流程。
2、配置管理：
    通过load_configs函数加载并合并多个 YAML 配置文件，实现模块化配置（如模型参数、数据参数、训练参数等），便于参数调优和版本管理。
3、目录与日志设置：
    setup_logging_dirs函数根据参数自动创建或复用日志目录（包含检查点、配置文件子目录），确保实验记录的规范性；configure_logger支持 Wandb、TensorBoard 等多种日志工具，便于实验监控。
4、训练回调配置：
    configure_callbacks函数设置模型检查点（按评估指标保存最优模型）、早停策略（防止过拟合）、配置保存等回调，自动化训练流程。
5、训练与测试流程：
    main函数作为入口，协调数据模块、模型的实例化，根据参数执行训练（支持从头训练或从检查点恢复）、测试流程，是连接数据、模型与训练逻辑的核心枢纽。
"""