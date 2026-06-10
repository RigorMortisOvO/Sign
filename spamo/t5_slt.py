import os  # 用于文件路径操作、环境变量设置
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # PyTorch神经网络模块
import random  # 用于随机操作（如帧序列裁剪）
import math  # 用于数学计算（如帧采样率计算）

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from typing import Dict, List, Optional, Tuple, Any  # 类型注解，提升代码可读性

import torch.nn.functional as F  # PyTorch常用函数（如激活函数、归一化）

from torch.nn.utils.rnn import pad_sequence  # 用于序列填充，统一批次内序列长度
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration  # T5模型及Tokenizer
from transformers import BertConfig, BertModel  # Bert配置与模型（此处未实际使用，预留扩展）
from peft import LoraConfig, get_peft_model, TaskType  # PEFT库，实现参数高效微调（LoRA）

from spamo.tconv import TemporalConv  # 自定义时序卷积模块，处理视觉特征的时间维度
from utils.helpers import create_mask, derangement  # 工具函数：生成序列掩码、打乱列表（错位排列）
from spamo.mm_projector import build_vision_projector  # 构建视觉特征投影器，对齐视觉与文本特征维度
# 新增：导入交叉注意力块（用于sent_fusions与视觉特征融合）
from spamo.mm_projector import CrossAttentionBlock
from utils.evaluate import evaluate_results  # 评估函数，计算BLEU、ROUGE等翻译指标
from spamo.clip_loss import clip_loss  # CLIP风格对比损失，用于跨模态对齐
from spamo.asb import AbstractSLT  # 手语翻译抽象基类，定义通用接口
from transformers import get_cosine_schedule_with_warmup  # 余弦学习率调度器（带热身阶段）

# 禁用tokenizers并行化，避免多进程环境下潜在死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置PyTorch浮点数矩阵乘法精度为"high"，平衡性能与精度
torch.set_float32_matmul_precision('high')


class FlanT5SLT(AbstractSLT):
    """
    基于FlanT5的手语翻译（Sign Language Translation, SLT）模型，支持多模态特征融合。
    核心功能：将手语视频的空间特征（静态视觉信息）和时空特征（动态运动信息）转换为自然语言文本。
    """

    def __init__(
            self,
            tuning_type: str = 'lora',  # 模型微调策略：'lora'（低秩适应，参数高效）或'freeze'（冻结T5主干）
            model_name: Optional[str] = None,  # FlanT5预训练模型名称/路径（如"google/flan-t5-xl"）
            frame_sample_rate: int = 1,  # 视频帧采样率（每隔N帧取1帧，降低计算量）
            prompt: str = '',  # 翻译提示词模板（如"Translate the sign language into {}"，{}填充目标语言）
            input_size: int = 1024,  # 输入视觉特征维度（如ViT提取的空间特征维度）
            fusion_mode: str = 'joint',  # 特征融合模式：'joint'（联合空间+时空）、'spatial'（仅空间）、'spatiotemporal'（仅时空）
            inter_hidden: int = 768,  # 视觉特征投影后的中间维度
            max_frame_len: int = 1024,  # 最大视觉帧序列长度（超过则随机裁剪）
            max_txt_len: int = 64,  # 文本序列最大长度（用于Tokenizer截断）
            cross_modal_align: bool = False,  # 是否启用视觉-文本跨模态对齐损失
            warm_up_steps: Optional[int] = None,  # 热身步数（前N步仅用对比损失训练，先对齐模态）
            combined_loss: bool = False,  # 是否使用联合损失（T5生成损失 + 对比损失）
            alpha: float = 0.1,  # 总损失中对比损失的权重(视觉-目标损失在总损失中的权重)
            gamma: float = 0.1,  # 视觉-上下文损失在总损失中的权重============================================================
            use_resampler: bool = False,  # 是否使用特征重采样器（当前版本未实现）
            sampling_length: int = 64,  # 重采样长度（未使用，预留参数）
            cache_dir: str = "/data3/models",  # 预训练模型缓存目录
            use_in_context: bool = False,  # 是否启用上下文学习（In-Context Learning）
            num_in_context: int = 0,  # 上下文示例数量（如3个跨语言平行句对）
            lora_r: int = 16,  # LoRA低秩矩阵的秩（控制可训练参数数量）
            lora_alpha: int = 32,  # LoRA缩放因子（alpha/r决定更新幅度）
            lora_dropout: float = 0.1,  # LoRA层的Dropout概率

            use_dynamic_weight: bool = True,  # 是否启用上下文动态权重======================================================
            weight_normalize: bool = True,  # 权重是否归一化==============================================================
            cross_attn_num_heads: Optional[int] = None,  # 新增：交叉注意力头数（默认与T5一致）

            word_top_k: int = 10,  # 单词路径：保留与视觉最相关的Top-K单词（默认3）

            word_top_k_range: Tuple[int, int] = (4, 6),  # 新增：动态Top-K范围（训练随机）
            # word_top_k_range: Tuple[int, int] = (3, 5),  # 新增：动态Top-K范围（训练随机）
            dropout_prob: float = 0.2,  # 新增：单词嵌入、Top-K特征的Dropout概率
            dropout_prob_sent: float = 0.1,  # 新增：单词嵌入、Top-K特征的Dropout概率

            **kwargs  # 传递给父类AbstractSLT的参数（如lr、monitor、beam_size等）
    ):
        super().__init__(**kwargs)  # 调用父类构造函数，初始化训练相关参数（学习率、监控指标等）

        # 存储模型配置参数（便于后续方法调用）
        self.input_size = input_size
        self.prompt = prompt
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha  # 视觉-目标文本损失权重
        self.gamma = gamma  # 视觉-上下文损失权重===========================================================================
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # 上下文权重计算参数===============================================================================================
        self.use_dynamic_weight = use_dynamic_weight
        self.weight_normalize = weight_normalize

        # 新增：交叉注意力相关参数存储
        self.cross_attn_num_heads = cross_attn_num_heads
        self.cross_attn_block = None  # 交叉注意力块（后续在prepare_models中初始化）

        self.word_top_k = word_top_k

        # 原有单词路径参数替换为动态范围（兼容旧代码，优先用范围）
        self.word_top_k_range = word_top_k_range
        self.word_top_k_min, self.word_top_k_max = word_top_k_range
        self.val_word_top_k = (self.word_top_k_min + self.word_top_k_max) // 2  # 验证固定K值（中间值）

        # 新增：Dropout层（训练时启用，验证时自动关闭）
        self.word_embed_dropout = nn.Dropout(dropout_prob)  # 单词嵌入后Dropout
        self.topk_feat_dropout = nn.Dropout(dropout_prob)  # Top-K特征后Dropout
        # self.sent_embed_dropout = nn.Dropout(dropout_prob_sent) # 三个句子嵌入融合前对每个句子嵌入的Dropout
        self.sent_spatial_dropout = nn.Dropout2d(dropout_prob_sent)  # 作用于通道维度，三个句子嵌入融合前对每个句子嵌入的Dropout

        self.prepare_models(model_name)  # 初始化文本模型（T5）、Tokenizer、视觉投影器等

        # 应用选定的微调策略
        if tuning_type == 'freeze':
            self._freeze_model()  # 冻结T5参数，仅训练视觉相关层
        elif tuning_type == 'lora':
            self._apply_lora()  # 为T5添加LoRA适配器，实现参数高效微调

        self.set_container()  # 初始化存储生成结果和参考文本的容器（用于评估）

    # def load_pretrained_weights(self, checkpoint_path: str) -> None:
    #     """Load weights from a pretrained checkpoint."""
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

    #     # Get model's state dict
    #     model_state_dict = self.state_dict()
    #     checkpoint_state_dict = checkpoint['state_dict']

    #     # Filter out mismatched keys
    #     filtered_state_dict = {}
    #     for k, v in checkpoint_state_dict.items():
    #         if k in model_state_dict and v.size() == model_state_dict[k].size():
    #             filtered_state_dict[k] = v

    #     # Load the filtered state dict
    #     self.load_state_dict(filtered_state_dict)
    #     print(f'Checkpoint loaded from {checkpoint_path}. Loaded {len(filtered_state_dict)}/{len(checkpoint_state_dict)} parameters.')

    def load_pretrained_weights(self, checkpoint_path):
        """从预训练检查点加载模型权重"""
        # 加载检查点，自动映射到当前设备（CPU/GPU）
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # 加载模型状态字典（包含权重参数）
        self.load_state_dict(checkpoint['state_dict'])
        print(f'已从 {checkpoint_path} 加载检查点.')

    def _apply_lora(self) -> None:
        """为T5模型应用LoRA（Low-Rank Adaptation）适配器，实现参数高效微调"""
        # 配置LoRA参数
        lora_config = LoraConfig(
            r=self.lora_r,  # 低秩矩阵的秩
            lora_alpha=self.lora_alpha,  # 缩放因子
            target_modules=["q", "v"],  # 仅对T5注意力层的Query和Value投影应用LoRA
            lora_dropout=self.lora_dropout,  # LoRA层的Dropout概率
            bias="none",  # 不微调模型偏置项
            task_type=TaskType.SEQ_2_SEQ_LM  # 任务类型：序列到序列语言生成
        )
        # 将T5模型包装为LoRA模型（仅新增低秩参数可训练）
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        print("已为T5模型应用LoRA适配器.")



    def _freeze_model(self) -> None:
        """冻结T5模型所有参数，仅训练视觉特征投影器、时序编码器等新增模块"""
        self.t5_model.eval()  # 将T5设为评估模式（关闭Dropout等训练特有层）
        # 遍历T5所有参数，设置为不可训练
        for params in self.t5_model.parameters():
            params.requires_grad = False
        print("T5模型已冻结.")

    def set_container(self) -> None:
        """初始化容器，用于存储验证/测试阶段的生成结果和参考文本（后续计算评估指标）"""
        self.generated = []  # 存储模型生成的翻译文本
        self.references = []  # 存储真实参考文本（标注的正确翻译）

    def prepare_models(self, t5_model: str) -> None:
        """
        准备文本模型、Tokenizer和视觉特征处理模块

        参数:
            t5_model: T5预训练模型的名称或本地路径
        """

        # 加载FlanT5序列到序列生成模型（用于手语文本翻译）
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model,
            cache_dir=self.cache_dir,  # 模型缓存目录（避免重复下载）
            torch_dtype=torch.bfloat16,  # 使用bfloat16精度，平衡显存占用与计算精度
        )

        # 加载T5对应的Tokenizer（用于文本的编码/解码）
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model,
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,  # 文本序列最大长度（超过则截断）
        )

        # 构建视觉特征投影器：将不同类型的视觉特征映射到统一维度
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)  # 空间特征投影（线性层）
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)  # 时空特征投影（线性层，输入维度1024）
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden,
                                                  self.t5_model.config.hidden_size)  # 融合后投影到T5隐藏层维度（2层MLP+GELU）

        # 加载时序编码器（1D卷积），处理视觉特征的时间维度相关性
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        # 跨模态对齐的温度参数（可学习），用于缩放视觉-文本相似度
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        # 新增：初始化交叉注意力块（用于sent_fusions与视觉特征的细粒度融合）
        # self.cross_attn_num_heads = self.cross_attn_num_heads or self.t5_model.config.num_attention_heads  # 默认复用T5的注意力头数
        self.cross_attn_num_heads = self.t5_model.config.num_attention_heads  # 默认复用T5的注意力头数
        self.cross_attn_block = CrossAttentionBlock(
            dim=self.t5_model.config.hidden_size,  # 输入维度=T5的hidden_size（与视觉/文本特征维度一致）
            num_heads=self.cross_attn_num_heads,  # 注意力头数（与T5一致，保证训练稳定性）
            qkv_bias=True,  # 启用QKV偏置（与T5配置对齐）
            mlp_ratio=4.0,  # MLP隐藏层比例（复用CrossAttentionBlock默认值）
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
        ).to(self.device)  # 移动到当前计算设备（GPU/CPU）
        # print(f"已初始化交叉注意力块，头数：{self.cross_attn_num_heads}")

    def prepare_inputs(
            self,
            visual_outputs: torch.Tensor,
            visual_mask: torch.Tensor,
            samples: Dict,
            split: str,
            batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """
        准备T5模型的输入：将视觉特征与文本提示（Prompt）拼接，生成模型可接受的嵌入格式

        参数:
            visual_outputs: 处理后的视觉特征张量（形状：[batch_size, seq_len, hidden_dim]）
            visual_mask: 视觉特征的掩码（1表示有效帧，0表示填充帧，形状：[batch_size, seq_len]）
            samples: 输入样本字典（包含文本、语言、上下文示例等信息）
            split: 当前数据分割（'train'/'val'/'test'）
            batch_idx: 当前批次索引

        返回:
            joint_outputs: 视觉特征+文本提示的拼接嵌入（形状：[batch_size, total_seq_len, hidden_dim]）
            joint_mask: 拼接后的掩码（形状：[batch_size, total_seq_len]）
            output_tokens: 目标文本的Tokenizer输出（含input_ids和attention_mask）
            targets: 目标文本标签（填充token替换为-100，避免计算损失）
        """
        bs = visual_outputs.shape[0]  # 批次大小

        # 生成带目标语言的提示词（如"Translate the sign language into German"）
        prompts = [f'{self.prompt}'] * bs  # 为每个样本复制提示词模板
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]  # 填充目标语言

        # 若启用上下文学习，将跨语言示例拼接到提示词后（如"Eng=德 法=德 ..."）
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]

        # 对提示词进行Tokenize（转换为模型可识别的ID序列）
        input_tokens = self.t5_tokenizer(
            prompts,
            padding="longest",  # 按批次内最长提示词填充
            truncation=True,  # 超过max_txt_len则截断
            return_tensors="pt",  # 返回PyTorch张量
        ).to(self.device)  # 移动到当前计算设备（GPU/CPU）

        # 计算视觉特征和提示词的有效长度（排除填充部分）
        visual_lengths = visual_mask.sum(1)  # 每个样本的有效视觉帧数
        prompt_lengths = input_tokens.attention_mask.sum(1)  # 每个样本的有效提示词Token数
        new_lengths = visual_lengths + prompt_lengths  # 拼接后的总序列长度

        # 将提示词的Token ID转换为嵌入向量（使用T5的词嵌入层）
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

        # 逐个样本拼接视觉特征和提示词嵌入（避免填充干扰有效特征）
        joint_outputs = []
        for i in range(bs):
            # 取当前样本的有效视觉特征（排除填充帧）
            vis_out = visual_outputs[i, :visual_lengths[i], :]
            # 取当前样本的有效提示词嵌入（排除填充Token）
            prompt_embeds = input_embeds[i, :prompt_lengths[i], :]
            # 拼接（视觉特征在前，提示词嵌入在后）
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)

        # 对拼接后的序列进行填充，统一批次内所有样本的长度
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        # 生成拼接序列的掩码（标记有效部分）
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)

        # 对目标文本（真实翻译结果）进行Tokenize
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        # 准备目标标签：将填充Token（pad_token_id）替换为-100（PyTorch会忽略-100的损失计算）
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据特征融合模式，处理空间特征和/或时空特征，生成统一格式的视觉输入

        参数:
            samples: 输入样本字典（包含空间特征、时空特征、长度信息等）

        返回:
            visual_outputs: 处理后的视觉特征张量（形状：[batch_size, seq_len, hidden_dim]）
            visual_masks: 视觉特征的掩码（形状：[batch_size, seq_len]）
        """
        # 根据融合模式，确定需要使用的视觉特征类型
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True  # 联合使用空间和时空特征
        else:
            spatial = self.fusion_mode == 'spatial'  # 仅使用空间特征
            spatiotemporal = self.fusion_mode == 'spatiotemporal'  # 仅使用时空特征

        # 处理空间特征（如ViT提取的静态帧特征）
        if spatial:
            # 对批次内的空间特征进行填充（统一序列长度）
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            # 将空间特征投影到中间维度（inter_hidden）
            spatial_outputs = self.spatio_proj(pixel_values)
            # 生成空间特征的掩码（标记有效帧）
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)

        # 处理时空特征（如VideoMAE提取的动态运动特征）
        if spatiotemporal:
            # 对批次内的时空特征进行填充
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            # 将时空特征投影到中间维度（inter_hidden）
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            # 生成时空特征的掩码
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)

        # 联合模式：融合空间特征和时空特征
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]  # 批次大小
            # 计算每个样本的有效空间/时空特征长度
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            new_length = spatial_length + spatiotemporal_length  # 融合后的总长度

            # 逐个样本拼接有效空间和时空特征（避免填充干扰）
            joint_outputs = []
            for i in range(bs):
                # 取当前样本的有效空间特征
                valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                # 取当前样本的有效时空特征
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                # 拼接两种特征
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                joint_outputs.append(concat_sample)
            # 填充拼接后的特征，统一批次长度
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)

            # 通过时序编码器处理融合特征的时间维度相关性
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0, 2, 1),  # 转换为[batch, channel, seq_len]，适配1D卷积输入
                torch.tensor(new_length.tolist(), device=self.device)  # 原始序列长度（用于计算卷积后长度）
            )

            # 调整特征维度为[batch, seq_len, channel]，生成最终掩码
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),  # 卷积后的有效长度
                device=self.device
            )
        else:
            # 单特征模式：仅使用空间或时空特征
            if spatial:
                # 对空间特征应用时序编码器，捕捉时间相关性
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0, 2, 1),  # 适配1D卷积
                    torch.tensor(samples['num_frames'], device=self.device)  # 原始帧数量
                )
                # 调整维度并生成掩码
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1, 0, 2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(),
                    device=self.device
                )
            elif spatiotemporal:
                # 时空特征直接使用（无需额外时序编码）
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError("无效的特征融合模式，需为'joint'/'spatial'/'spatiotemporal'")

        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        """
        处理数据加载器返回的原始批次数据，转换为模型可接受的结构化字典

        参数:
            batch: 原始批次数据（列表形式，每个元素为单个样本的字典）

        返回:
            结构化输入字典（包含处理后的视觉特征、文本、元数据等）
        """
        # 初始化存储列表
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []  # 存储上下文学习的跨语言示例

        context_sentences_list = []  # 存储上下文三句列表 [batch, 3]=======================================================

        max_frame_len = self.max_frame_len  # 视觉帧序列的最大长度（超过则裁剪）

        # 逐个处理批次中的样本
        for sample in batch:
            # 仅处理包含有效空间特征的样本（排除空特征）
            if sample['pixel_value'].shape[0] != 0:
                # 计算采样后的帧数（按frame_sample_rate间隔取帧）
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]  # 按采样率取帧

                # 收集样本元数据
                ids.append(sample['id'])  # 样本唯一ID
                texts.append(sample['text'].lower())  # 目标文本（统一小写，避免大小写干扰）
                glosses.append(sample['gloss'])  # 手语词汇标注（Gloss，可选）
                langs.append(sample['lang'])  # 目标语言（如"German"）

                # 收集上下文句子（s1/s2/s3）==============================================================================
                context_sentences_list.append(sample.get('context_sentences', [['', '', '']]))

                # 跨语言示例（英文=目标语、法文=目标语、西班牙文=目标语）
                _ex_lang_trans = [
                    f"{sample['en_text']}={sample['text']}",
                    f"{sample['fr_text']}={sample['text']}",
                    f"{sample['es_text']}={sample['text']}"
                ]
                _ex_lang_trans = _ex_lang_trans[:self.num_in_context]  # 截取指定数量的示例
                ex_lang_translations.append(' '.join(_ex_lang_trans))  # 拼接为单个字符串

                # 若采样后的帧数超过最大长度，随机裁剪到max_frame_len
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]

                # 存储处理后的空间特征及长度
                num_frames.append(nframe)
                pixel_values.append(pval)

                # 处理时空特征（glor_value）
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        # 若时空特征为列表（多尺度特征），拼接为单个张量
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))  # 总长度
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))  # 单尺度特征长度

        # 打乱跨语言示例（避免示例与样本自身强关联，防止过拟合）
        ex_lang_translations = derangement(ex_lang_translations)

        # 返回结构化输入字典
        return {
            'pixel_values': pixel_values,  # 处理后的空间特征列表
            'glor_values': glor_values,  # 处理后的时空特征列表
            'bool_mask_pos': masks,  # 预留掩码（未使用）
            'ids': ids,  # 样本ID列表
            'text': texts,  # 目标文本列表
            'ex_lang_trans': ex_lang_translations,  # 上下文示例列表
            'gloss': glosses,  # Gloss列表
            'lang': langs,  # 目标语言列表
            'num_frames': num_frames,  # 空间特征帧数列表
            'glor_lengths': glor_lengths,  # 时空特征长度列表

            'context_sentences': context_sentences_list,  # 传递上下文句子================================================
        }

    def visual_textual_align(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor,
                             samples: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        独立计算两个跨模态对比损失（批量优化版）：
        1. loss_target：视觉特征与目标文本的对比损失
        2. loss_context：视觉特征与上下文句子的对比损失（按动态权重加权）
        核心优化：将所有上下文句子（batch_size×3个）批量Tokenize和编码，减少循环计算
        返回：loss_target, loss_context, sent_fusions（新增返回上下文融合特征）
        """
        device = visual_outputs.device
        batch_size = visual_outputs.shape[0]  # 批次大小


        # ================================== 视觉特征处理（不变）==================================
        # 1. 计算视觉特征全局表示（用于两种损失的统一输入）
        visual_mask = visual_masks.unsqueeze(-1).float()  # [batch, seq_len, 1]：扩展维度便于广播
        # 全局平均池化：(特征×掩码)求和后除以有效长度（避免填充影响）
        visual_embeds = (visual_outputs * visual_mask).sum(dim=1) / visual_mask.sum(dim=1).clamp(
            min=1e-8)  # [batch, dim]
        visual_embeds = F.normalize(visual_embeds, dim=-1)  # L2归一化，确保相似度计算有效

        # ================================== 损失1：视觉-目标文本对比损失（不变）==================================
        # 1. 目标文本Tokenize（批量处理）
        target_tokens = self.t5_tokenizer(
            samples['text'],  # 目标文本列表：[batch]
            padding="longest",  # 按批次内最长文本填充
            truncation=True,
            return_tensors="pt"  # 返回PyTorch张量
        ).to(device)  # 移动到计算设备

        # 2. 目标文本全局表示（词嵌入→掩码池化→归一化）
        target_embeds = self.t5_model.encoder.embed_tokens(target_tokens.input_ids)  # [batch, seq_len, dim=2048]：词嵌入
        # print("target_embeds形状:", target_embeds.shape)
        target_mask = target_tokens.attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]：文本掩码
        # 掩码全局平均池化（排除填充部分）
        target_embeds = (target_embeds * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp(
            min=1e-8)  # [batch, dim]
        target_embeds = F.normalize(target_embeds, dim=-1)  # L2归一化

        # 3. 计算CLIP风格对比损失
        # 目标文本与视觉特征的相似度矩阵：[batch, batch]（对角线为正样本）
        logits_per_target = torch.matmul(target_embeds, visual_embeds.t()) * self.logit_scale.exp() # 温度参数（可学习）
        loss_target = clip_loss(logits_per_target)  # 目标损失（已批次平均）

        # ================================== 损失2：视觉-上下文句子对比损失（批量优化）==================================
        # 1. 准备上下文句子数据
        context_sentences_list = samples['context_sentences']  # 上下文句子列表：[batch, 3]（每个样本3个句子）

        # 2. 收集所有上下文句子（扁平化处理）
        # 从[batch, 3]转为[batch×3]，例如：batch=2时，[s1_0,s2_0,s3_0, s1_1,s2_1,s3_1]
        all_sents = [sent for sample_sents in context_sentences_list for sent in sample_sents]  # 长度=batch_size×3

        # ================================== 批量计算上下文句子嵌入==================================
        # 3. 批量Tokenize所有上下文句子（一次调用处理所有句子）
        sent_tokens_batch = self.t5_tokenizer(
            all_sents,  # 所有句子：[batch×3]
            padding="longest",  # 按批次内最长文本填充（得到max_seq_len）
            truncation=True,
            return_tensors="pt"  # 返回张量
        ).to(device)  # 输出：input_ids[batch×3, max_seq_len]；attention_mask[batch×3, max_seq_len]

        # 4. 批量编码为“单词级向量”（保留每个单词的嵌入，不做句子池化）
        # 输入：input_ids[batch×3, max_seq_len] → 输出：sent_word_embeds[batch×3, max_seq_len, dim]
        sent_word_embeds = self.t5_model.encoder.embed_tokens(sent_tokens_batch.input_ids)  # 单词级向量，未池化

        # ------------------------------ 路径A：单词路径（动态Top-K：30%-40% 随机范围 + 四舍五入）------------------------------

        # 1. 计算单词与视觉的相似度
        visual_embeds_repeated = visual_embeds.repeat_interleave(3, dim=0)  # [batch×3, dim]
        # [batch×3, max_seq_len]
        word_sim = F.cosine_similarity(sent_word_embeds, visual_embeds_repeated.unsqueeze(1), dim=2)

        # 把填充位置的相似度置负无穷：确保填充位置的相似度极低，避免被TopK选中
        word_sim = word_sim.masked_fill(sent_tokens_batch.attention_mask == 0, -1e9)

        # 2. 计算每个样本动态的 K 值 (Batch×3 个独立的 K)
        # 获取每个句子的有效 token 数量 [batch×3]
        valid_lens = sent_tokens_batch.attention_mask.sum(dim=1).float()

        # --- 修改开始：计算 30%-40% 的范围 (使用四舍五入) ---

        # 【核心修改】：使用 torch.round 进行四舍五入，然后转为 long 类型
        # clamp(min=1) 确保即使是很短的句子 (如 len=1, 1*0.3=0.3->0) 也能至少选出 1 个词
        min_k = torch.round(valid_lens * 0.3).long().clamp(min=1)
        max_k = torch.round(valid_lens * 0.4).long().clamp(min=1)

        if self.training:
            # 【训练阶段】：在 [min_k, max_k] 范围内随机选择
            # 计算范围差值
            diff = max_k - min_k

            # 生成随机偏移量：
            # 1. torch.rand_like 生成 [0, 1) 的随机浮点数
            # 2. 乘以 (diff + 1) 得到 [0, diff + 1)
            # 3. floor() 向下取整得到 [0, diff] 的整数偏移
            random_offsets = (torch.rand_like(valid_lens) * (diff + 1).float()).floor().long()

            # 最终 K 值 = 下界 + 随机偏移
            k_values = min_k + random_offsets
        else:
            # 【验证/测试阶段】：取中间值 35% 并四舍五入，保证结果确定性
            k_values = torch.round(valid_lens * 0.35).long().clamp(min=1)

        # 双重保险：确保 K 值不会超过句子本身的有效长度 (防止 40% 四舍五入后越界，虽然数学上不太可能)
        k_values = torch.min(k_values, valid_lens.long())

        # --- 修改结束 ---

        # 3. GPU 并行化：取当前批次中最大的 K 值作为统一宽度的容器
        max_k = k_values.max().item()
        # 防止 max_k 超过实际 Tensor 的最大物理长度 (max_seq_len)
        max_k = min(max_k, word_sim.shape[1])

        # 4. 初步选出 Top-Max_K (包含多余的词)
        # top_sim: [batch×3, max_k], top_indices: [batch×3, max_k]
        top_sim, top_indices = word_sim.topk(max_k, dim=1)

        # 5. 构建动态掩码，过滤掉超过各自 k_values 的部分
        # 生成一个 [1, max_k] 的序列: [0, 1, 2, ..., max_k-1]
        range_vector = torch.arange(max_k, device=device).unsqueeze(0)
        # mask: [batch×3, max_k]，如果位置索引 < 样本各自的k_value 则为 True (有效)
        mask_k = range_vector < k_values.unsqueeze(1)

        # 将不需要的位置（超过各自 K 的部分）相似度置为负无穷，Softmax 后权重变为 0
        top_sim_masked = top_sim.masked_fill(~mask_k, -1e9)
        # [batch×3, max_k]
        top_word_weights = F.softmax(top_sim_masked, dim=1)

        # 6. 提取对应的特征
        # [batch×3, max_k, dim]
        top_indices_exp = top_indices.unsqueeze(-1).repeat(1, 1, self.t5_model.config.hidden_size)
        top_word_feats = torch.gather(sent_word_embeds, dim=1, index=top_indices_exp)

        # 训练时应用 Dropout (可选)
        # if self.training:
        #     top_word_feats = self.topk_feat_dropout(top_word_feats)

        # 7. 计算权重并融合
        # 加权求和: [batch×3, max_k, dim] * [batch×3, max_k, 1] -> sum -> [batch×3, dim]
        # 无效部分的权重为 0，不会影响 sum 结果
        sent_word_feat = (top_word_feats * top_word_weights.unsqueeze(-1)).sum(dim=1)

        # 重组形状
        sent_word_feat_per_sample = sent_word_feat.reshape(batch_size, 3, self.t5_model.config.hidden_size)

        # # # ------------------------------ 路径A：单词路径（动态Top-K视觉相关单词）------------------------------
        # # 4. 批量编码为“单词级向量”（保留每个单词的嵌入，不做句子池化）
        # # sent_word_embeds = self.t5_model.encoder.embed_tokens(sent_tokens_batch.input_ids)  # [batch×3, max_seq_len, dim]
        # # 新增：单词嵌入后应用Dropout（训练时生效）
        # # sent_word_embeds = self.word_embed_dropout(sent_word_embeds)
        #
        # visual_embeds_repeated = visual_embeds.repeat_interleave(3, dim=0)  # [batch×3, dim]
        # word_sim = F.cosine_similarity(sent_word_embeds, visual_embeds_repeated.unsqueeze(1),dim=2)  # [batch×3, max_seq_len]
        # word_sim = word_sim * sent_tokens_batch.attention_mask.float()  # 过滤填充词
        #
        # # 新增：动态选择Top-K值（根据模型训练状态自动切换）
        # if self.training:
        #     # 训练时：从[min, max]随机选K（确保不超过单词数量）
        #     current_k = random.randint(self.word_top_k_min, self.word_top_k_max)
        #     current_k = min(current_k, word_sim.shape[1])
        #     # current_k = min(10, word_sim.shape[1])
        # else:
        #     # 验证/测试时：使用固定中间值K
        #     current_k = min(self.val_word_top_k, word_sim.shape[1])
        #     # current_k = min(random.choice([9,10]), word_sim.shape[1])
        #     # current_k = min(4, word_sim.shape[1])
        #     # current_k = word_sim.shape[1]//2
        #     # print(f"TopK={word_sim.shape[1]}")
        #     # valid_token_counts = sent_tokens_batch.attention_mask.sum(dim=1)  # 每个样本的有效 token 数
        #     # min_count = valid_token_counts.min()
        #     # mean_count = valid_token_counts.float().mean()
        #     # counts = torch.bincount(valid_token_counts)
        #     # # b. 找到出现次数最多的值的索引，这个索引就是众数
        #     # mode_count = counts.argmax()
        #     # print(f"有效Token数={valid_token_counts}")
        #     # print(f"最小Token数={min_count}")
        #     # print(f"平均Token数={mean_count}")
        #     # print(f"众数Token数={mode_count}")
        #     # print(f"TopK={word_sim.shape[1]}")
        #     # print(f"TopK={current_k}")
        #
        # # 筛选Top-K单词（用动态current_k替代原有固定self.word_top_k）
        # top_word_indices = word_sim.topk(current_k, dim=1).indices  # [batch×3, current_k]
        # top_word_indices_exp = top_word_indices.unsqueeze(-1).repeat(1, 1, self.t5_model.config.hidden_size)
        # top_word_feats = torch.gather(sent_word_embeds, dim=1, index=top_word_indices_exp)  # [batch×3, current_k, dim]
        #
        # # 新增：Top-K特征后应用Dropout（训练时生效）
        # # top_word_feats = self.topk_feat_dropout(top_word_feats)
        #
        # # 计算Top-K单词权重并融合（原有逻辑不变）
        # top_word_weights = F.softmax(torch.gather(word_sim, dim=1, index=top_word_indices),dim=1)  # [batch×3, current_k]
        # sent_word_feat = (top_word_feats * top_word_weights.unsqueeze(-1)).sum(dim=1)  # [batch×3, dim]
        # sent_word_feat_per_sample = sent_word_feat.reshape(batch_size, 3,self.t5_model.config.hidden_size)  # [batch, 3, dim]

        # -------------------------- 核心修改：按权重融合（替代全局平均） --------------------------
        # # 1. 计算每个句子与视觉特征的相似度（衡量句子相关性）
        # # visual_embeds形状：[batch, dim]，扩展为[batch, 1, dim]以匹配句子特征维度
        # sent_visual_sim = F.cosine_similarity(sent_word_feat_per_sample, visual_embeds.unsqueeze(1),dim=2)  # [batch, 3]  每个句子的相似度
        # # 2. 对相似度做softmax，得到句子权重（高相关句子权重更高）
        # sent_weights = F.softmax(sent_visual_sim, dim=1)  # [batch, 3]，权重和为1
        # # 3. 按权重融合3个句子的特征（加权求和）
        # word_path_feat = (sent_word_feat_per_sample * sent_weights.unsqueeze(-1)).sum(dim=1)  # [batch, dim]
        # --------------------------------------------------------------------------------------

        word_path_feat = sent_word_feat_per_sample.mean(dim=1)  # [batch, dim]
        word_path_feat = F.normalize(word_path_feat, dim=-1)

        # # ------------------------------ 路径A：单词路径（Top-K视觉相关单词）------------------------------
        # visual_embeds_repeated = visual_embeds.repeat_interleave(3, dim=0)  # [batch×3, dim]
        # word_sim = F.cosine_similarity(sent_word_embeds, visual_embeds_repeated.unsqueeze(1), dim=2)  # [batch×3, max_seq_len]
        # word_sim = word_sim * sent_tokens_batch.attention_mask.float()  # 过滤填充词
        # top_k = min(self.word_top_k, word_sim.shape[1])
        # top_word_indices = word_sim.topk(top_k, dim=1).indices  # [batch×3, top_k]
        # top_word_indices_exp = top_word_indices.unsqueeze(-1).repeat(1, 1, self.t5_model.config.hidden_size)
        # top_word_feats = torch.gather(sent_word_embeds, dim=1, index=top_word_indices_exp)  # [batch×3, top_k, dim]
        # top_word_weights = F.softmax(torch.gather(word_sim, dim=1, index=top_word_indices), dim=1)  # [batch×3, top_k]
        #
        # # Top-K单词融合得到sent_word_feat
        # sent_word_feat = (top_word_feats * top_word_weights.unsqueeze(-1)).sum(dim=1)  # [batch×3, dim]
        # sent_word_feat_per_sample = sent_word_feat.reshape(batch_size, 3, self.t5_model.config.hidden_size)  # [batch, 3, dim]
        #
        # # # 方法一：权重平均
        # word_path_feat = sent_word_feat_per_sample.mean(dim=1)  # [batch, dim]
        # word_path_feat = F.normalize(word_path_feat, dim=-1)
        #
        # # # 方法二：用句子与视觉的相似度作为权重（替代平均）
        # # sent_weights = F.softmax(sent_word_feat_per_sample, dim=1).unsqueeze(-1)  # [batch, 3, 1]
        # # word_path_feat = (sent_word_feat_per_sample * sent_weights).sum(dim=1)  # 加权求和

        # ------------------------------ 路径B：句子路径（按相似度权重融合所有句子）------------------------------
        # 1. 生成每个句子的整体嵌入（词嵌入+掩码平均池化）
        sent_mask_exp = sent_tokens_batch.attention_mask.unsqueeze(-1).float()  # [batch×3, max_seq_len, 1]
        sent_embeds = (sent_word_embeds * sent_mask_exp).sum(dim=1) / sent_mask_exp.sum(dim=1).clamp(min=1e-8)  # [batch×3, dim]
        sent_embeds = F.normalize(sent_embeds, dim=-1)
        # 2. 计算每个句子与视觉全局特征的相似度（视觉锚定）
        sent_sim = F.cosine_similarity(sent_embeds, visual_embeds_repeated, dim=1)  # [batch×3]

        # -------------------------------------------------------------------------------------------
        # 2. 设定阈值，过滤相似度低于阈值的句子（视为无效噪声句）
        # valid_mask = (sent_sim > 0.2).float()  # 仅保留相似度>0.2的句子
        # sent_embeds = sent_embeds * valid_mask.unsqueeze(-1)  # 无效句特征置为0

        # # 3. 对过滤后的句子嵌入应用标准Dropout（随机丢弃单个元素）
        # sent_embeds = self.sent_embed_dropout(sent_embeds)  # 此时丢弃的主要是有效句中的噪声特征

        # 空间Dropout（按通道丢弃）
        # 应用：[batch×3, dim] → [batch×3, dim, 1]（扩展为2D）→ 丢弃通道 → 恢复形状
        # sent_embeds = self.sent_spatial_dropout(sent_embeds.unsqueeze(-1)).squeeze(-1)
        # ----------------------------------------------------------------------------------------------

        # 3. 重组为样本维度：[batch×3] → [batch, 3]（每个样本的3个句子相似度）
        sent_sim_per_sample = sent_sim.reshape(batch_size, 3)  # [batch, 3]
        # 4. 直接对3个句子的相似度进行归一化
        sent_weights = F.softmax(sent_sim_per_sample, dim=1)  # [batch, 3]：每个句子的权重（和为1）
        # 5. 所有句子按权重融合 → 句子路径最终特征
        sent_embeds_per_sample = sent_embeds.reshape(batch_size, 3, self.t5_model.config.hidden_size)  # [batch, 3, dim]
        sent_path_feat = (sent_embeds_per_sample * sent_weights.unsqueeze(-1)).sum(dim=1)  # [batch, dim]（加权求和）
        sent_path_feat = F.normalize(sent_path_feat, dim=-1)  # 归一化

        # ------------------------------ A+B双路径动态融合（视觉引导权重）------------------------------
        word_path_sim = F.cosine_similarity(word_path_feat, visual_embeds, dim=1)  # [batch]
        sent_path_sim = F.cosine_similarity(sent_path_feat, visual_embeds, dim=1)  # [batch]
        path_weights = F.softmax(torch.stack([word_path_sim, sent_path_sim], dim=1), dim=1)  # [batch, 2]
        word_path_weight = path_weights[:, 0].unsqueeze(1)  # [batch, 1]
        sent_path_weight = path_weights[:, 1].unsqueeze(1)  # [batch, 1]
        sent_fusions = (word_path_weight * word_path_feat) + (sent_path_weight * sent_path_feat)  # [batch, dim]
        sent_fusions = F.normalize(sent_fusions, dim=-1)

        # ===============================================================================================================

        # ----------------------------基于groundtruth的上下文与视觉特征对齐约束-------------------------------------------
        # 5. 构建“单词-目标”映射：每个句子的所有单词对应同一个目标文本向量
        # 5.1 先按原逻辑得到句子级目标映射：[batch, dim] → [batch×3, dim]（每个句子对应一个目标）
        target_embeds_repeated_3 = target_embeds.repeat_interleave(3, dim=0)  # [batch×3, dim]
        # 5.2 扩展到单词维度：[batch×3, dim] → [batch×3, max_seq_len, dim]（每个单词对应同一个句子的目标）
        # 逻辑：每个句子的max_seq_len个单词，共享同一个目标文本向量
        target_embeds_word_repeated = target_embeds_repeated_3.unsqueeze(1).repeat(1, sent_word_embeds.shape[1], 1)  # [batch×3, max_seq_len, dim]

        # 6. 计算“单词与目标文本的相似度”（单词权重）
        # 输入：sent_word_embeds[batch×3, max_seq_len, dim] vs target_embeds_word_repeated[batch×3, max_seq_len, dim]
        # 输出：word_weights[batch×3, max_seq_len]（每个单词与目标文本的相似度）
        word_weights = F.cosine_similarity(sent_word_embeds, target_embeds_word_repeated, dim=2)  # dim=2：按特征维度计算相似度

        # 权重归一化
        if self.weight_normalize:
            # 7. 单词权重归一化（排除填充，按句子内有效单词的相似度比例归一化）
            # 7.1 用句子掩码过滤填充单词：填充位置（mask=0）的权重直接置为0（不参与后续计算）
            word_weights = word_weights * sent_tokens_batch.attention_mask.float()  # [batch×3, max_seq_len]
            # 7.2 计算每个句子的有效单词相似度总和（排除填充），并做比例归一化
            # 计算总和（dim=1：按句子内单词求和），clamp避免除以0（当句子全为填充时）
            sum_weights = word_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [batch×3, 1]
            # 比例归一化：每个有效单词的权重 = 自身相似度 / 句子内有效单词相似度总和
            word_weights = word_weights / sum_weights  # [batch×3, max_seq_len]
        else:
            # 7. 单词权重归一化（排除填充，按句子内单词归一化）
            # 7.1 用句子掩码过滤填充单词：填充位置（mask=0）的权重置为-1e9（softmax后接近0，不影响）
            word_weights = word_weights * sent_tokens_batch.attention_mask.float() + (1.0 - sent_tokens_batch.attention_mask.float()) * (-1e9)
            # 7.2 按句子内单词归一化（dim=1）：确保每个句子的单词权重和为1
            word_weights = F.softmax(word_weights, dim=1)  # [batch×3, max_seq_len]

        # 8. 生成“单词加权的句子特征”（替代原代码的全局平均句子特征）
        # 8.1 权重扩展维度：[batch×3, max_seq_len] → [batch×3, max_seq_len, 1]（便于广播相乘）
        word_weights_expanded = word_weights.unsqueeze(-1)  # [batch×3, max_seq_len, 1]
        # 8.2 单词加权求和：每个单词向量 × 对应权重 → 按句子内单词求和（dim=1）
        # 输入：sent_word_embeds[batch×3, max_seq_len, dim] × word_weights_expanded[batch×3, max_seq_len, 1]
        # 输出：sent_embeds[batch×3, dim]（单词加权后的句子级特征）
        sent_embeds_clip = (sent_word_embeds * word_weights_expanded).sum(dim=1)  # [batch×3, dim]
        # 8.3 L2归一化（与原代码一致，确保相似度计算尺度统一）
        sent_embeds_clip = F.normalize(sent_embeds_clip, dim=-1)  # [batch×3, dim]

        # ================================== 批量计算上下文句子权重（动态权重）==================================
        # 9. 批量计算“句子与目标文本的相似度”（句子权重，仍用句子级特征计算）
        # 输入：sent_embeds[batch×3, dim]（单词加权的句子特征） vs target_embeds_repeated_3[batch×3, dim]
        # 输出：weights[batch×3]（每个句子与所属目标的相似度）
        sent_weights = F.cosine_similarity(sent_embeds_clip, target_embeds_repeated_3, dim=1)
        # 重组为[batch, 3]（按样本分组，每组3个句子权重）
        sent_weights = sent_weights.reshape(batch_size, 3)  # [batch, 3]

        # 10. 句子权重归一化
        if self.weight_normalize:
            sent_weights = sent_weights / sent_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [batch, 3]：简单比例归一化
        else:
            sent_weights = F.softmax(sent_weights, dim=1)  # [batch, 3]：突出高相关句子

        # ================================== 批量计算上下文损失==================================
        # 初始化上下文损失
        loss_context = torch.tensor(0.0, device=device)

        # 12. 句子嵌入按样本重组
        sent_embeds_per_sample_clip = sent_embeds_clip.reshape(batch_size, 3, -1)  # [batch, 3, dim]

        # 13. 批量融合所有样本的上下文句子（单词加权的句子特征参与融合）
        sent_weights_expanded_clip = sent_weights.unsqueeze(-1)  # [batch, 3, 1]：权重扩展维度
        sent_fusions_clip = (sent_embeds_per_sample_clip * sent_weights_expanded_clip).sum(dim=1)  # [batch, dim]：每个样本的三句融合特征
        sent_fusions_clip = F.normalize(sent_fusions_clip, dim=-1)

        # 14. 计算文本-视觉相似度矩阵（CLIP风格）
        logits_per_fusion = torch.matmul(sent_fusions_clip, visual_embeds.t()) * self.logit_scale.exp()  # [batch, batch]

        # 15. 计算上下文损失
        loss_context = clip_loss(logits_per_fusion)  # 批次平均损失

        # 返回两个独立损失 + sent_fusions（新增上下文融合特征）
        return loss_target, loss_context, sent_fusions

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        训练、验证、测试阶段的共享逻辑：计算模型输出和损失，记录日志
        训练/验证/测试共享逻辑：
        - 分别计算生成损失、视觉-目标损失、视觉-上下文损失
        - 总损失 = 翻译文本生成损失 + alpha*视觉-目标损失 + gamma*视觉-上下文损失（三者独立）

        参数:
            inputs: 结构化输入字典（来自get_inputs方法）
            split: 当前数据分割（'train'/'val'/'test'）
            batch_idx: 当前批次索引

        返回:
            loss: 模型损失值（标量张量）
            log_dict: 日志字典（包含损失、评估指标等）
        """
        # 步骤1：处理视觉输入，投影到与T5隐藏层匹配的维度
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs_t5_dim = self.fusion_proj(visual_outputs)  # [batch, seq_len_vis, t5_hidden_size]：视觉特征映射到T5维度

        # 初始化日志字典（存储损失和评估指标）
        log_dict = {}

        # ====================== 关键修改：前文组选择逻辑（训练随机选，验证/测试固定选）======================
        context_sentences_multi_groups = inputs['context_sentences']  # 格式：[batch, [group1, group2, ...]]，每组是[s1,s2,s3]
        context_sentences_groups_num = len(context_sentences_multi_groups)
        selected_context = []  # 存储最终选中的单组前文：[batch, [s1,s2,s3]]

        for i in range(context_sentences_groups_num):
            # 获取当前样本的所有前文组（如2组）
            groups = context_sentences_multi_groups[i]
            # 确保groups是列表格式，避免格式错误
            if not isinstance(groups, list) or len(groups) == 0:
                # 无有效前文组时，用空字符串填充默认组
                print("无有效前文组，用空字符串填充默认组")
                selected_group = ['', '', '']
            else:
                if split == 'train':
                    # 训练时：随机选择1组前文
                    idx = random.randint(0, len(groups) - 1)
                    selected_group = groups[idx]
                else:
                    # 验证/测试时：固定选择第1组（索引0），保证评估结果可复现
                    selected_group = groups[0]
            # 确保选中的组是3个句子，不足则补空字符串（适配后续处理逻辑）
            selected_group = selected_group[:3] + [''] * (3 - len(selected_group))
            selected_context.append(selected_group)

        # 替换inputs中的context_sentences为选中的单组前文，后续逻辑完全复用原有代码
        inputs['context_sentences'] = selected_context
        # print(f"选中前文内容为：{selected_context}\n\n\n")
        # ==============================================================================================

        # 步骤2：根据训练模式计算损失
        if self.cross_modal_align:

            # 模式1：启用跨模态对齐损失
            # 子模式1.1：纯对比学习（无生成损失，仅对齐模态）
            if self.warm_up_steps is None and not self.combined_loss:
                # 冻结T5模型，仅计算对比损失（不更新T5参数）
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs_t5_dim, visual_masks, inputs, split, batch_idx
                    )

                # 关键：拆分元组（获取两个独立损失 + sent_fusions）
                loss_target, loss_context, sent_fusions = self.visual_textual_align(visual_outputs_t5_dim, visual_masks, inputs)
                # 分别记录两个损失（单个张量）
                log_dict[f"{split}/Pure_comparison_target_loss"] = loss_target.item()  # 视觉-目标损失
                log_dict[f"{split}/Pure_comparison_context_loss"] = loss_context.item()  # 视觉-上下文损失
                loss = loss_target + loss_context
                log_dict[f"{split}/Pure_comparison_total_loss"] = loss.item()  # 记录总损失

            # 子模式1.2：热身阶段（前warm_up_steps步，仅用对比损失对齐模态）
            elif self.warm_up_steps is not None and self.global_step < self.warm_up_steps:
                # 冻结T5，仅训练视觉相关层以对齐模态
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs_t5_dim, visual_masks, inputs, split, batch_idx
                    )

                # 只用目标对比损失来热身（同时获取sent_fusions，不参与损失计算）
                loss_target, loss_context, sent_fusions = self.visual_textual_align(visual_outputs_t5_dim, visual_masks, inputs)
                log_dict[f"{split}/warm_target_loss"] = loss_target.item()
                loss = loss_target
                log_dict[f"{split}/total_loss"] = loss.item()  # 记录总损失

            # 子模式1.3：联合损失（生成损失 + 目标对比损失 + 上下文对比损失）
            else:
                # 步骤3：计算跨模态对齐损失 + 获取上下文融合特征sent_fusions
                loss_target, loss_context, sent_fusions = self.visual_textual_align(visual_outputs_t5_dim, visual_masks, inputs)

                # 新增：交叉注意力融合（视觉特征 ↔ sent_fusions）
                # Q=视觉特征（序列数据，每个帧对应一个特征），K/V=sent_fusions（上下文语义，扩展为序列长度=1）
                sent_fusions_expanded = sent_fusions.unsqueeze(1)  # [batch, 1, t5_hidden_size]：扩展为序列格式
                visual_outputs_fused = self.cross_attn_block(q=visual_outputs_t5_dim, x=sent_fusions_expanded)  # 交叉注意力交互

                # 步骤4：准备T5输入（融合后的视觉特征 + 提示词）
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    visual_outputs_fused, visual_masks, inputs, split, batch_idx
                )

                # T5模型前向传播（文本生成任务）
                outputs = self.t5_model(
                    inputs_embeds=input_embeds,  # 输入嵌入（融合后视觉特征+Prompt）
                    attention_mask=input_masks,  # 输入掩码（忽略填充）
                    decoder_attention_mask=output_tokens.attention_mask,  # 解码器掩码
                    labels=targets,  # 目标标签（用于计算生成损失）
                    output_hidden_states=True,  # 输出隐藏状态（预留扩展）
                    return_dict=True  # 返回字典格式输出
                )

                # 计算翻译文本生成损失+目标文本跨模态对比损失+上下文跨模态对比损失
                t5_loss = outputs.loss  # T5生成损失（交叉熵损失）

                # 总损失 = 翻译生成损失 + alpha*视觉-目标损失 + gamma*视觉-上下文损失
                loss = t5_loss + self.alpha * loss_target + self.gamma * loss_context

                # 记录总损失相关指标
                log_dict[f"{split}/combine_generation_loss"] = t5_loss
                log_dict[f"{split}/combine_target_loss"] = loss_target
                log_dict[f"{split}/combine_context_loss"] = loss_context
                log_dict[f"{split}/total_loss"] = loss
        else:
            # 模式2：标准生成模式（仅计算T5生成损失）
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs_t5_dim, visual_masks, inputs, split, batch_idx
            )

            # T5模型前向传播
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=output_tokens.attention_mask,
                labels=targets,
                output_hidden_states=True,
                return_dict=True
            )
            loss = outputs.loss  # 总损失=生成损失
            log_dict[f"{split}/pure_generation_loss"] = loss  # 记录生成损失

        # 步骤2：验证/测试阶段：生成文本并存储结果（用于后续评估）
        if split != "train":
            if split == "test":
                # # 确保 cross_attn_block 可用（即使未启用跨模态对齐，也强制计算融合特征）
                _, _, sent_fusions = self.visual_textual_align(visual_outputs_t5_dim, visual_masks, inputs)
                # 用交叉注意力融合视觉特征和上下文特征
                sent_fusions_expanded = sent_fusions.unsqueeze(1)  # [batch, 1, t5_hidden_size]
                visual_outputs_fused = self.cross_attn_block(q=visual_outputs_t5_dim, x=sent_fusions_expanded)
                gen_visual_outputs = visual_outputs_fused  # 强制使用融合特征

                # gen_visual_outputs = visual_outputs_t5_dim

            else:
                # print("验证阶段")
                # 准备生成用的输入（融合后的视觉特征+提示词，或原始视觉特征+提示词）
                if self.cross_modal_align and self.warm_up_steps is not None and self.global_step >= self.warm_up_steps:
                    # 联合损失模式：使用融合后的视觉特征
                    gen_visual_outputs = visual_outputs_fused
                else:
                    # 其他模式：使用原始映射后的视觉特征
                    gen_visual_outputs = visual_outputs_t5_dim

            input_embeds, input_masks, _, _ = self.prepare_inputs(
                gen_visual_outputs, visual_masks, inputs, split, batch_idx
            )

            # 模型生成文本（使用束搜索提升生成质量）
            generated = self.t5_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                num_beams=5,  # 束搜索宽度=5
                max_length=self.max_txt_len,  # 生成文本最大长度
                top_p=0.9,  # 采样Top-p=0.9（平衡多样性与准确性）
                do_sample=True,  # 启用采样（非确定性生成）
            )

            # 解码生成结果和参考文本（将Token ID转换为字符串，跳过特殊Token）
            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]  # 统一小写

            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]  # 统一小写

            # 存储生成结果和参考文本（用于epoch结束时计算评估指标）
            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)

            # （注释代码）实时计算评估指标（当前移至epoch结束时统一计算）
            # eval_res = evaluate_results(
            #     predictions=generated_strings,
            #     references=reference_strings,
            #     split=split,
            #     tokenizer='zh' if inputs['lang'][0] == 'Chinese' else '13a',
            #     device=self.device
            # )
            # log_dict.update(eval_res)

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        """
        PyTorch Lightning钩子函数：验证epoch结束时执行
        功能：打印生成示例、计算评估指标、记录日志、重置结果容器
        """
        # 打印前5个验证样本的生成结果与参考文本（带颜色区分）
        # print("\n===== 验证集示例 =====")
        # for i in range(min(5, len(self.generated))):
        #     print(f"\033[94m参考文本: {self.references[i]}\033[0m")  # 蓝色：参考文本
        #     print(f"\033[92m生成文本: {self.generated[i]}\033[0m")  # 绿色：生成文本
        #     print("-" * 50)  # 分隔线

        # 计算验证集评估指标（BLEU、ROUGE等）
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',  # 数据分割标记为验证集
            # tokenizer='zh' if inputs['lang'][0] == 'Chinese' else '13a',  # （预留）中文/英文分词适配
            device=self.device
        )

        # ========== 每次验证都打印评估结果 ==========
        # print("\n===== 验证集评估结果 =====")
        for metric_name, metric_value in eval_res.items():
            # 格式化输出（保留4位小数，适配BLEU/ROUGE等指标）
            print(f"\n验证集评估结果{metric_name}: {metric_value:.4f}\n")
        # ======================================================

        # 将评估指标记录到日志（支持多GPU分布式训练同步）
        self.log_dict(eval_res, sync_dist=True)

        # 重置结果容器（为下一个epoch做准备）
        self.set_container()

    def on_test_epoch_end(self) -> None:
        """
        PyTorch Lightning钩子函数：测试epoch结束时执行
        功能：打印生成示例、计算测试指标、记录日志、重置结果容器
        """
        # # 打印前5个测试样本的生成结果与参考文本（注：原文打印"Validation Examples"为笔误，功能为测试集）
        # print("\n===== 测试集示例 =====")
        # for i in range(min(5, len(self.generated))):
        #     print(f"\033[94m参考文本: {self.references[i]}\033[0m")  # 蓝色：参考文本
        #     print(f"\033[92m生成文本: {self.generated[i]}\033[0m")  # 绿色：生成文本
        #     print("-" * 50)  # 分隔线

        # 计算测试集评估指标（BLEU、ROUGE等）
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',  # 数据分割标记为测试集
            device=self.device
        )

        # 将测试指标记录到日志（支持分布式同步）
        self.log_dict(eval_res, sync_dist=True)
        # 重置结果容器
        self.set_container()


    # def configure_optimizers(self):
    #     """
    #     配置优化器和学习率调度器（PyTorch Lightning要求实现）
    #     返回：优化器与调度器的字典配置
    #     """
    #     # 初始化AdamW优化器（适合Transformer模型的优化器）
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.lr,  # 学习率（从父类继承）
    #         eps=1e-8,  # 数值稳定性参数
    #         weight_decay=0.01,  # 权重衰减（正则化，防止过拟合）
    #         betas=(0.9, 0.98)  # 动量参数（常用Transformer优化配置）
    #     )
    #
    #     # 计算总训练步数（用于调度器）
    #     if hasattr(self.trainer, 'estimated_stepping_batches'):
    #         # 优先使用Trainer估算的总步数（适配动态批次等场景）
    #         total_steps = self.trainer.estimated_stepping_batches
    #     else:
    #         #  fallback计算：总步数 = epochs × 每epoch批次 × 梯度累积系数倒数
    #         max_epochs = self.trainer.max_epochs  # 最大训练epoch数
    #         train_dataloader = self.trainer.train_dataloader  # 训练数据加载器
    #         # 适配DataLoaderWrapper（如分布式场景）
    #         if hasattr(train_dataloader, 'dataloader'):
    #             train_dataloader = train_dataloader.dataloader
    #
    #         batches_per_epoch = len(train_dataloader)  # 每epoch的批次数
    #         total_steps = batches_per_epoch * max_epochs  # 无梯度累积的总步数
    #
    #         # 若启用梯度累积，总步数需除以累积批次系数
    #         if hasattr(self.trainer, 'accumulate_grad_batches'):
    #             total_steps = total_steps // self.trainer.accumulate_grad_batches
    #
    #     # 计算热身步数（总步数的10%，避免初始学习率过高导致训练不稳定）
    #     warmup_steps = int(total_steps * 0.12)
    #
    #     # 初始化余弦学习率调度器（带热身阶段）
    #     scheduler = get_cosine_schedule_with_warmup(
    #         optimizer=optimizer,
    #         num_warmup_steps=warmup_steps,  # 热身步数
    #         num_training_steps=total_steps,  # 总训练步数
    #     )
    #
    #     # 返回优化器与调度器配置（调度器按步骤更新）
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",  # 调度器更新间隔：每步更新
    #             "frequency": 1,  # 每1步更新一次
    #         },
    #     }
    #


    def configure_optimizers(self):
        # 1. 初始化AdamW优化器
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            eps=1e-8,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )

        # 2. 计算总训练步数
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            max_epochs = self.trainer.max_epochs
            train_dataloader = self.trainer.train_dataloader
            if hasattr(train_dataloader, 'dataloader'):
                train_dataloader = train_dataloader.dataloader
            batches_per_epoch = len(train_dataloader)
            total_steps = batches_per_epoch * max_epochs
            if hasattr(self.trainer, 'accumulate_grad_batches'):
                total_steps = total_steps // self.trainer.accumulate_grad_batches

        # 3. 固定热身步数与最小学习率
        # warmup_steps = 12000
        warmup_steps = int(total_steps * 0.5)
        min_lr = 0.00020


        # 4. 自定义学习率调度逻辑
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # 线性热身阶段
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # 余弦衰减阶段
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr / self.lr + (1 - min_lr / self.lr) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # 5. 返回优化器与调度器配置
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }