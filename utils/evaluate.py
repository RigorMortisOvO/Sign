from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF, TER


def evaluate_results(predictions, references, split="train", device='cpu', tokenizer='13a'):
    """
    使用BLEU和ROUGE指标评估预测结果的质量。
    主要用于自然语言生成任务（如手语翻译）中，衡量模型生成文本与参考文本的匹配程度。

    参数:
        predictions (list): 模型生成的预测序列列表（如翻译结果）
        references (list): 参考序列列表（如人工标注的正确翻译）
        split (str): 当前评估的数据分割（如"train"训练集、"test"测试集）
        device (str): 计算设备（此处未实际使用，保留参数兼容）
        tokenizer (str): 分词方式，默认'13a'为sacrebleu中常用的分词器

    返回:
        dict: 包含各评估指标分数的字典
    """
    # 初始化存储评估分数的字典
    log_dicts = {}

    # 计算BLEU-4分数（4-gram匹配度），适用于衡量翻译流畅性和准确性
    # BLEU指标通过比较n-gram重叠度评估生成文本质量，max_ngram_order=4即计算到4-gram
    bleu4 = BLEU(max_ngram_order=4, tokenize=tokenizer).corpus_score(predictions, [references]).score
    # 将BLEU-4分数存入字典，键名包含数据分割标识
    log_dicts[f"{split}/bleu4"] = bleu4

    # 若为测试集，计算更详细的评估指标
    if split == 'test':
        # 计算BLEU-1到BLEU-3分数，分别对应1-gram到3-gram的匹配度
        # 低阶n-gram（如1-gram）反映词汇准确性，高阶n-gram（如3-gram）反映句法流畅性
        for i in range(1, 4):
            score = BLEU(max_ngram_order=i, tokenize=tokenizer).corpus_score(predictions, [references]).score
            log_dicts[f"{split}/bleu" + str(i)] = score

        # 初始化ROUGE评分器，计算ROUGE-L指标（基于最长公共子序列LCS）
        # ROUGE-L更关注长序列匹配，适合评估结构相似性
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)  # use_stemmer启用词干提取，增强词汇匹配鲁棒性

        # 对每个预测-参考对计算ROUGE-L分数
        rouge_scores = [scorer.score(ref, pred)['rougeL'] for ref, pred in zip(references, predictions)]

        # 聚合ROUGE-L分数：计算平均精度、召回率和F1分数
        # 精度（precision）：预测中匹配参考的比例；召回率（recall）：参考中被预测匹配的比例
        # F1分数：精度和召回率的调和平均，综合两者表现
        avg_precision = sum(score.precision for score in rouge_scores) / len(rouge_scores)
        avg_recall = sum(score.recall for score in rouge_scores) / len(rouge_scores)
        avg_f1 = sum(score.fmeasure for score in rouge_scores) / len(rouge_scores)

        # 将ROUGE-L的各项指标存入字典
        log_dicts[f"{split}/rougeL_precision"] = avg_precision
        log_dicts[f"{split}/rougeL_recall"] = avg_recall
        log_dicts[f"{split}/rougeL_f1"] = avg_f1

    # 返回包含所有评估指标的字典
    return log_dicts


"""
该代码实现了一个用于评估文本生成结果质量的函数evaluate_results，主要应用于手语翻译等自然语言生成任务。其核心功能是通过计算BLEU和ROUGE-L两类主流指标，量化模型生成文本与参考文本的匹配程度，具体说明如下：
1、BLEU（Bilingual Evaluation Understudy）：
    基于 n-gram（n 元语法）的重叠度计算，评估生成文本与参考文本的词汇和短句匹配度。
    函数中对训练集默认计算 BLEU-4（4-gram），对测试集额外计算 BLEU-1 到 BLEU-3，全面反映从单词语法到短句结构的匹配质量。
2、ROUGE-L（Recall-Oriented Understudy for Gisting Evaluation - L）：
    基于最长公共子序列（LCS）计算，更关注长序列的结构匹配，适合评估生成文本的整体语义连贯性。
    函数中计算其精度（precision）、召回率（recall）和 F1 分数，综合衡量生成文本的准确性和完整性。
"""