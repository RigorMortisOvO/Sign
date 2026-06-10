import numpy as np


class LambdaWarmUpCosineScheduler:
    """
    带预热阶段的余弦退火学习率调度器
    注意：使用时基础学习率（base_lr）应设为1.0，实际学习率为该调度器的输出乘以base_lr
    """

    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        """
        初始化调度器参数

        参数:
            warm_up_steps: 预热步数，在该阶段学习率从lr_start线性增长到lr_max
            lr_min: 学习率的最小值（衰减阶段的最终值）
            lr_max: 学习率的最大值（预热结束时达到的值）
            lr_start: 预热阶段的初始学习率
            max_decay_steps: 总的衰减步数（包含预热阶段）
            verbosity_interval: 日志打印间隔，若大于0，则每间隔该步数打印一次学习率信息
        """
        self.lr_warm_up_steps = warm_up_steps  # 预热步数
        self.lr_start = lr_start  # 预热初始学习率
        self.lr_min = lr_min  # 最小学习率
        self.lr_max = lr_max  # 最大学习率
        self.lr_max_decay_steps = max_decay_steps  # 总衰减步数
        self.last_lr = 0.  # 记录上一次的学习率
        self.verbosity_interval = verbosity_interval  # 日志间隔

    def schedule(self, n, **kwargs):
        """
        计算第n步的学习率乘数

        参数:
            n: 当前训练步数
            **kwargs: 额外参数（未使用）

        返回:
            第n步的学习率乘数
        """
        # 打印日志（若设置了间隔）
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"当前步骤: {n}, 最近学习率乘数: {self.last_lr}")

        # 预热阶段：学习率从lr_start线性增长到lr_max
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        # 衰减阶段：使用余弦函数从lr_max衰减到lr_min
        else:
            # 计算衰减比例（t在[0,1]之间）
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)  # 确保t不超过1.0
            # 余弦退火公式：lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*t))
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        """让调度器实例可直接调用，返回第n步的学习率乘数"""
        return self.schedule(n, **kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    支持多轮循环的带预热余弦退火学习率调度器
    每轮循环可配置不同的参数，适用于需要阶段性调整学习率策略的场景
    注意：使用时基础学习率（base_lr）应设为1.0，实际学习率为该调度器的输出乘以base_lr
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        """
        初始化多轮调度器参数

        参数:
            warm_up_steps: 列表，每个元素表示对应循环的预热步数
            f_min: 列表，每个元素表示对应循环的最小学习率乘数
            f_max: 列表，每个元素表示对应循环的最大学习率乘数（预热结束时的值）
            f_start: 列表，每个元素表示对应循环的预热初始学习率乘数
            cycle_lengths: 列表，每个元素表示对应循环的总步数
            verbosity_interval: 日志打印间隔
        """
        # 确保所有参数列表长度一致（每个循环对应一组参数）
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps  # 各循环预热步数列表
        self.f_start = f_start  # 各循环预热初始乘数列表
        self.f_min = f_min  # 各循环最小乘数列表
        self.f_max = f_max  # 各循环最大乘数列表
        self.cycle_lengths = cycle_lengths  # 各循环总步数列表
        # 计算累计循环步数（用于确定当前步骤处于哪个循环）
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.  # 记录上一次的学习率乘数
        self.verbosity_interval = verbosity_interval  # 日志间隔

    def find_in_interval(self, n):
        """
        确定当前步骤n属于哪个循环阶段

        参数:
            n: 当前训练步数

        返回:
            循环索引（第几个循环）
        """
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1
        return interval  # 若超出所有循环，返回最后一个循环

    def schedule(self, n, **kwargs):
        """
        计算第n步的学习率乘数（多循环版本）

        参数:
            n: 当前训练步数
            **kwargs: 额外参数（未使用）

        返回:
            第n步的学习率乘数
        """
        # 确定当前所在的循环
        cycle = self.find_in_interval(n)
        # 计算当前循环内的相对步数（减去之前循环的总步数）
        n_in_cycle = n - self.cum_cycles[cycle]

        # 打印日志（若设置了间隔）
        if self.verbosity_interval > 0:
            if n_in_cycle % self.verbosity_interval == 0:
                print(f"当前步骤: {n}, 最近学习率乘数: {self.last_f}, 当前循环: {cycle}")

        # 当前循环的预热阶段：线性增长
        if n_in_cycle < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n_in_cycle + self.f_start[
                cycle]
            self.last_f = f
            return f
        # 当前循环的衰减阶段：余弦退火
        else:
            # 计算当前循环内衰减比例（t在[0,1]之间）
            t = (n_in_cycle - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            # 余弦退火公式
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        """让调度器实例可直接调用，返回第n步的学习率乘数"""
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    """
    支持多轮循环的带预热线性衰减学习率调度器
    继承自LambdaWarmUpCosineScheduler2，与余弦退火的区别是衰减阶段使用线性衰减
    """

    def schedule(self, n, **kwargs):
        """
        计算第n步的学习率乘数（线性衰减版本）

        参数:
            n: 当前训练步数
            **kwargs: 额外参数（未使用）

        返回:
            第n步的学习率乘数
        """
        # 确定当前所在的循环（复用父类方法）
        cycle = self.find_in_interval(n)
        # 计算当前循环内的相对步数
        n_in_cycle = n - self.cum_cycles[cycle]

        # 打印日志（若设置了间隔）
        if self.verbosity_interval > 0:
            if n_in_cycle % self.verbosity_interval == 0:
                print(f"当前步骤: {n}, 最近学习率乘数: {self.last_f}, 当前循环: {cycle}")

        # 预热阶段：与父类一致，线性增长
        if n_in_cycle < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n_in_cycle + self.f_start[
                cycle]
            self.last_f = f
            return f
        # 衰减阶段：线性衰减（替代父类的余弦退火）
        else:
            # 线性衰减公式：从f_max线性降至f_min
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n_in_cycle) / \
                self.cycle_lengths[cycle]
            self.last_f = f
            return f


"""
该代码实现了三种学习率调度器，用于在模型训练过程中动态调整学习率，以优化模型收敛效果。核心作用是通过 “预热 + 衰减” 的策略，解决训练初期学习率过高导致的模型不稳定问题，同时在训练后期通过衰减学习率帮助模型收敛到更优解。具体说明如下：
1、LambdaWarmUpCosineScheduler
    基础的单循环调度器，包含两个阶段：
        预热阶段：学习率从初始值lr_start线性增长到最大值lr_max（共warm_up_steps步），避免初始学习率过高对模型的冲击。
        余弦衰减阶段：预热结束后，学习率通过余弦函数从lr_max平滑衰减到lr_min（总步数为max_decay_steps），余弦衰减相比线性衰减更温和，有助于模型在后期精细调整参数。
2、LambdaWarmUpCosineScheduler2
    支持多轮循环的余弦退火调度器，适用于需要阶段性调整学习率策略的场景（如多阶段训练）。每轮循环可独立配置预热步数、最大 / 最小学习率等参数，通过累计循环步数自动判断当前所处阶段，实现多轮 “预热 + 余弦衰减” 的交替。
3、LambdaLinearScheduler
    继承自多轮调度器，与余弦版本的区别是衰减阶段采用线性衰减（学习率从f_max线性降至f_min），适用于需要更快速降低学习率的场景。
"""