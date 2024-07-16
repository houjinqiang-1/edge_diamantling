import random
from typing import List, Tuple
from mvc_env import MvcEnv
'''
Data 类:

表示一个经验数据点。
包含关于图 (g)、当前状态序列 (s_t)、下一个状态序列 (s_prime)、采取的动作 (a_t)、获得的奖励 (r_t) 以及一个表示状态是否终止的标志 (term_t) 的信息。

LeafResult 类:
表示从 SumTree 中检索叶子节点的结果。
包含叶子节点的索引 (leaf_idx)、优先级值 (p) 以及对应的 Data 对象。

SumTree 类:
实现了 SumTree 数据结构，用于优先级经验回放。   区间分成多个子区间，每个区间构建一个二叉树
存储经验及其优先级，允许根据优先级有效地进行采样。
提供了添加具有给定优先级的数据、更新优先级以及根据随机值获取叶子节点的方法。

ReplaySample 类:
表示从回放内存中采样的一批经验。
存储了各种组件的列表，例如图对象 (g_list)、状态序列 (list_st)、下一个状态序列 (list_s_primes)、动作 (list_at)、奖励 (list_rt) 和终止状态标志 (list_term)。

Memory 类:
管理深度强化学习中使用的回放内存。
利用 SumTree 实现了优先级经验回放。
存储经验及其优先级。
提供了存储新经验、从环境中添加具有 N 步回报的经验、采样一批经验以及根据误差更新优先级的方法。
'''

class Data:
    def __init__(self):
        """
        Data对象的构造函数。
        """
        self.g = None  # 存储图对象
        self.s_t = []  # 存储状态序列
        self.s_prime = []  # 存储下一个状态序列
        self.a_t = 0  # 存储动作
        self.r_t = 0.0  # 存储奖励
        self.term_t = False  # 存储终止状态标志

# 纯Python版本的LeafResult对象
class LeafResult:
    def __init__(self):
        """
        LeafResult对象的构造函数。
        """
        self.leaf_idx = 0
        self.p = 0.0
        self.data = None  # 存储Data对象

# 纯Python版本的SumTree对象
class SumTree:
    def __init__(self, capacity: int):
        """
        SumTree对象的构造函数。

        参数：
        - capacity: SumTree的容量。
        """
        self.capacity = capacity
        self.data_pointer = 0
        self.minElement = float("inf")
        self.maxElement = 0.0
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity

    def add(self, p: float, data: Data):
        """
        向SumTree中添加数据。

        参数：
        - p: 数据的优先级。
        - data: 要添加的Data对象。
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx: int, p: float):
        """
        更新SumTree中的优先级。

        参数：
        - tree_idx: 要更新的节点索引。
        - p: 更新后的优先级。
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        if p < self.minElement:
            self.minElement = p

        if p > self.maxElement:
            self.maxElement = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float) -> LeafResult:
        """
        获取SumTree中的叶子节点。

        参数：
        - v: 随机值，用于选择叶子节点。

        返回：
        LeafResult对象，包含叶子节点的信息。
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        result = LeafResult()
        result.leaf_idx = leaf_idx
        result.p = self.tree[leaf_idx]
        result.data = self.data[data_idx]
        return result

# 纯Python版本的ReplaySample对象
class ReplaySample:
    def __init__(self, batch_size: int):
        """
        ReplaySample对象的构造函数。

        参数：
        - batch_size: 采样的批量大小。
        """
        self.b_idx = [0] * batch_size
        self.ISWeights = [0.0] * batch_size
        self.g_list = []  # 存储图对象列表
        self.list_st = []  # 存储状态序列列表
        self.list_s_primes = []  # 存储下一个状态序列列表
        self.list_at = []  # 存储动作序列
        self.list_rt = []  # 存储奖励序列
        self.list_term = []  # 存储终止状态标志序列

# 纯Python版本的Memory对象
class Memory:
    def __init__(self, epsilon: float, alpha: float, beta: float, beta_increment_per_sampling: float, abs_err_upper: float, capacity: int):
        """
        Memory对象的构造函数。

        参数：
        - epsilon: 用于优先级更新的小值。
        - alpha: 优先级采样指数。
        - beta: 重要性采样指数。
        - beta_increment_per_sampling: 重要性采样指数的增量。
        - abs_err_upper: 优先级的上限。
        - capacity: Memory的容量。
        """
        self.tree = SumTree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = abs_err_upper

    def store(self, transition: Data):
        """
        向Memory中存储数据。

        参数：
        - transition: 要存储的Data对象。
        """
        max_p = self.tree.maxElement
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def add(self, env: MvcEnv, n_step: int):
        """
        向Memory中添加MvcEnv的经验。

        参数：
        - env: MvcEnv对象。
        - n_step: N步回报的步数。
        """
        assert env.isTerminal()
        num_steps = len(env.state_seq)
        assert num_steps > 0

        env.sum_rewards[num_steps - 1] = env.reward_seq[num_steps - 1]
        for i in range(num_steps - 1, -1, -1):
            if i < num_steps - 1:
                env.sum_rewards[i] = env.sum_rewards[i + 1] + env.reward_seq[i]

        for i in range(num_steps):
            term_t = False
            cur_r = 0.0
            s_prime = []
            if i + n_step >= num_steps:
                cur_r = env.sum_rewards[i]
                s_prime = env.action_list.copy()
                term_t = True
            else:
                cur_r = env.sum_rewards[i] - env.sum_rewards[i + n_step]
                s_prime = env.state_seq[i + n_step].copy()

            transition = Data()
            transition.g = env.graph
            transition.s_t = env.state_seq[i].copy()
            transition.a_t = env.act_seq[i]
            transition.r_t = cur_r
            transition.s_prime = s_prime.copy()
            transition.term_t = term_t

            self.store(transition)

    def sampling(self, batch_size: int) -> ReplaySample:
        """
        从Memory中采样一批数据。

        参数：
        - batch_size: 采样的批量大小。

        返回：
        ReplaySample对象，包含采样的数据。
        """
        result = ReplaySample(batch_size)
        total_p = self.tree.tree[0]
        pri_seg = total_p / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        min_prob = self.tree.minElement / total_p

        for i in range(batch_size):
            a = pri_seg * i
            b = pri_seg * (i + 1)
            v = random.uniform(a, b)
            leaf_result = self.tree.get_leaf(v)
            result.b_idx[i] = leaf_result.leaf_idx
            prob = leaf_result.p / total_p
            result.ISWeights[i] = (prob / min_prob) ** -self.beta
            result.g_list.append(leaf_result.data.g)
            result.list_st.append(leaf_result.data.s_t)
            result.list_s_primes.append(leaf_result.data.s_prime)
            result.list_at.append(leaf_result.data.a_t)
            result.list_rt.append(leaf_result.data.r_t)
            result.list_term.append(leaf_result.data.term_t)

        return result

    def batch_update(self, tree_idx: List[int], abs_errors: List[float]):
        """
        批量更新Memory中数据的优先级。

        参数：
        - tree_idx: 要更新的节点索引列表。
        - abs_errors: 更新后的优先级。
        """
        for i in range(len(tree_idx)):
            abs_errors[i] += self.epsilon
            clipped_error = min(abs_errors[i], self.abs_err_upper)
            ps = clipped_error ** self.alpha
            self.tree.update(tree_idx[i], ps)
