import os

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback


class LoggingCallback(Callback):
    """
    日志回调类，用于在模型测试和验证阶段记录生成的文本结果，便于后续分析模型性能。
    继承自PyTorch Lightning的Callback，可自动融入训练流程。
    """

    def __init__(self, **kwargs):
        super().__init__()

    def log_generated_text(
            self,
            save_dir,
            ids,
            vis_strings,
            gloss_strings,
            generated_strings,
            reference_strings,
            prefix=None
    ):
        """
        将生成的文本结果（包括参考文本、生成文本等）保存到文件中。

        参数:
            save_dir: 保存文件的根目录
            ids: 样本的唯一标识符列表
            vis_strings: 视觉特征相关的字符串表示列表
            gloss_strings: 手语 Gloss（手语词汇）字符串列表（可为空）
            generated_strings: 模型生成的文本列表
            reference_strings: 参考文本（真实标签）列表
            prefix: 文件名前缀（用于区分不同阶段，如验证/测试）
        """
        # 创建文本保存目录（在save_dir下的text子目录）
        save_dir = os.path.join(save_dir, "text")
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在，不存在则创建
        file_name = f"outputs.txt"  # 默认文件名

        # 如果指定了前缀，用前缀修改文件名（如"val-outputs.txt"）
        if prefix is not None:
            file_name = f"{prefix}-outputs.txt"

        # 如果存在gloss_strings（手语词汇），写入包含gloss的内容
        if gloss_strings != []:
            with open(os.path.join(save_dir, file_name), "w") as file:
                for id, vis, gls, gen, ref in zip(ids, vis_strings, gloss_strings, generated_strings,
                                                  reference_strings):
                    file.write(f"ID: {id}\nVis Token: {vis}\nGloss: {gls}\nReference: {ref}\nGenerated: {gen}\n\n")
        # 否则写入不包含gloss的内容
        else:
            with open(os.path.join(save_dir, file_name), "w") as file:
                for id, vis, gen, ref in zip(ids, vis_strings, generated_strings, reference_strings):
                    file.write(f"ID: {id}\nVis Token: {vis}\nReference: {ref}\nGenerated: {gen}\n\n")

    def on_test_end(self, trainer, pl_module):
        """
        测试阶段结束时自动调用的方法，用于触发文本日志记录。
        从模型中获取测试相关数据，并调用log_generated_text保存。

        参数:
            trainer: PyTorch Lightning的Trainer实例
            pl_module: 训练的模型（LightningModule实例）
        """
        # 从模型中获取测试样本的ID、视觉字符串、gloss、生成文本和参考文本
        ids = pl_module.id_list
        vis_strings = pl_module.vis_string_list
        glosses = pl_module.gloss_list
        generated = pl_module.generated_text_list
        references = pl_module.reference_text_list

        # 调用日志记录方法，保存测试结果
        self.log_generated_text(
            pl_module.logger.save_dir,  # 日志保存目录（从logger中获取）
            ids,
            vis_strings,
            glosses,
            generated,
            references,
        )


class SetupCallback(Callback):
    """
    初始化回调类，用于在训练开始前设置目录结构、保存配置文件，并在发生异常时保存检查点。
    确保训练过程的可复现性和稳定性。
    """

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        """
        初始化设置回调类。

        参数:
            resume: 是否从检查点恢复训练
            now: 当前时间戳（用于命名文件）
            logdir: 日志根目录
            ckptdir: 检查点保存目录
            cfgdir: 配置文件保存目录
            config: 项目配置（模型、数据等）
            lightning_config: PyTorch Lightning相关配置（训练器等）
        """
        super().__init__()

        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        """
        当训练过程中发生异常时调用，保存最后一个检查点，避免训练成果丢失。

        参数:
            trainer: Trainer实例
            pl_module: 模型实例
            exception: 发生的异常
        """
        # 仅在主进程（global_rank=0）执行，避免多进程重复操作
        if trainer.global_rank == 0:
            # 只有当训练已经进行了一定步数（非初始状态）时才保存
            if pl_module.global_step != 0:
                print("[INFO] 检测到异常，程序终止")
                # print("[INFO] 检测到异常，保存最后检查点...")
                # ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
                # trainer.save_checkpoint(ckpt_path)

    def on_train_start(self, trainer, pl_module):
        """
        训练开始时调用，创建必要的目录，并保存配置文件（确保实验可复现）。

        参数:
            trainer: Trainer实例
            pl_module: 模型实例
        """
        # 仅在主进程执行
        if trainer.global_rank == 0:
            # 创建日志、检查点、配置文件的保存目录（不存在则创建）
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # 保存项目配置（模型、数据等参数）到yaml文件
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, f"{self.now}-project.yaml"))

            # 保存Lightning配置（训练器参数等）到yaml文件
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"))


"""
该代码定义了两个 PyTorch Lightning 回调类（LoggingCallback和SetupCallback），用于辅助手语翻译模型（SpaMo）的训练、测试流程，主要作用是日志管理和训练环境初始化，确保实验的可追踪性和稳定性。
1. LoggingCallback类
    核心功能：记录模型在测试阶段生成的文本结果，便于后续分析模型翻译效果。
    关键方法：
        log_generated_text：将样本 ID、视觉特征标识、手语 Gloss（可选）、模型生成文本、参考文本（真实标签）等信息写入文件，保存路径为{save_dir}/text/outputs.txt（可通过前缀区分阶段）。
        on_test_end：在测试结束时自动触发，从模型中提取测试数据（如生成文本、参考文本），调用log_generated_text完成日志保存。
    作用：通过保存生成结果，方便开发者对比模型输出与真实标签，评估翻译质量（如人工检查或自动指标计算）。
2. SetupCallback类
    核心功能：初始化训练环境，保存配置文件，并在异常时保存检查点，保障实验的可复现性和安全性。
    关键方法：
        on_train_start：训练开始时创建日志目录（logdir）、检查点目录（ckptdir）、配置目录（cfgdir），并将项目配置（模型、数据参数）和 Lightning 配置（训练器参数）保存为 YAML 文件，确保实验参数可追溯。
        on_exception：训练过程中发生异常时，在主进程保存最后一个检查点（last.ckpt），避免因错误导致训练进度丢失。
    作用：标准化实验环境，保存关键配置，减少因参数丢失导致的复现问题；异常处理机制提高了训练的鲁棒性。
"""