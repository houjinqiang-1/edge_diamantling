# mvc_env.py
from typing import List, Set
import random
from disjoint_set import DisjointSet
from graph import Graph
import networkx as nx
import Mcc
class MvcEnv:
    def __init__(self, norm):
        # MvcEnv类的构造方法，初始化实例变量
        self.norm = norm  # 正则化参数
        self.graph = Graph(0)
        self.numCoveredEdges = 0  # 覆盖的边的数量
        self.CcNum = 1.0  # 连通分量数
        self.state_seq = []  # 存储状态序列
        self.act_seq = []  # 存储动作序列
        self.action_list = []  # 存储动作列表
        self.reward_seq = []  # 存储奖励序列
        self.sum_rewards = []  # 存储累积奖励
        # self.covered_set = set()  # 存储覆盖的节点集合
        self.avail_list = []  # 存储可用节点列表
        # self.single_nodes_after_act = set()
        self.prepare_remove_edges = set()
        # self.not_chose_nodes = set()
        self.remove_edge = [set(),set()]
        self.MaxCCList = [1]
        self.score = 0.0
        self.score_list = []
    def s0(self, _g: Graph):
        # 重置环境状态的方法
        self.graph = _g  # 使用新的图
        # self.covered_set.clear()  # 清空覆盖的节点集合
        self.action_list.clear()  # 清空动作列表
        self.numCoveredEdges = 0
        self.CcNum = 1.0  # 重置连通分量数
        self.state_seq.clear()  # 清空状态序列
        self.act_seq.clear()  # 清空动作序列
        self.reward_seq.clear()  # 清空奖励序列
        self.sum_rewards.clear()  # 清空累积奖励
        # self.single_nodes_after_act.clear()
        self.prepare_remove_edges.clear()
        self.avail_list.clear()
        self.remove_edge[0].clear()
        self.remove_edge[1].clear()
        self.MaxCCList = [1]
        self.score_list.clear()
        self.score = 0.0
        self.getMaxConnectedNodesNum()
    def step(self, a):
        # 执行一个动作的方法
        assert self.graph
        assert a not in self.prepare_remove_edges
        self.state_seq.append(self.action_list.copy())  # 将当前动作列表加入状态序列
        self.act_seq.append(a)  # 将当前动作加入动作序列
        # self.covered_set.add(a)  # 将动作对应的节点添加到覆盖集合
        self.action_list.append(a)  # 将动作添加到动作列表
        self.prepare_remove_edges.add(a)
        self.numCoveredEdges = 0
        r_t = self.getReward()  # 获取奖励

        for i in self.graph.edge_sequence[0]:
            if i in self.prepare_remove_edges:
                self.numCoveredEdges += 1

        self.reward_seq.append(r_t)  # 将奖励加入奖励序列
        self.sum_rewards.append(r_t)  # 将奖励加入累积奖励               
        return r_t

    def stepWithoutReward(self, a):
        # 执行一个动作但不计算奖励的方法
        assert self.graph
        assert a not in self.prepare_remove_edges
        self.prepare_remove_edges.add(a)  # 将动作对应的节点添加到覆盖集合
        self.action_list.append(a)  # 将动作添加到动作列表
        for i in self.graph.edge_sequence[0]:
            if i in self.prepare_remove_edges:
                self.numCoveredEdges += 1

        r_t = self.getReward()  # 获取奖励
        self.score += -1 * r_t
        self.MaxCCList.append(-1 * r_t * (self.graph.num_edges[0] + self.graph.num_edges[1]))
        self.numCoveredEdges = 0
        self.score_list.append(self.score)

    def randomAction(self):
        # 随机选择一个动作的方法
        assert self.graph
        self.avail_list = []
        for edge_idx in self.graph.edge_sequence[0]:
            if edge_idx not in self.prepare_remove_edges:
                self.avail_list.append(edge_idx)
        assert self.avail_list
        idx_1 = random.choice(self.avail_list)  # 从可用节点列表中随机选择一个动作
        return idx_1


    def isTerminal(self):
        # 判断是否达到终止状态的方法
        assert self.graph
        # return self.graph.num_edges[0] == (self.numCoveredEdges[0] + len(self.remove_edge[0])/2) or self.graph.num_edges[1] == (self.numCoveredEdges[1] + len(self.remove_edge[1])/2)
        return (self.graph.num_edges[0] + self.graph.num_edges[1] == self.numCoveredEdges) or \
               (self.graph.num_edges[0] == sum(1 for x in self.prepare_remove_edges if x < self.graph.num_edges[0])) or \
               (self.graph.num_edges[1] == sum(1 for x in self.prepare_remove_edges if x > self.graph.num_edges[0]))

        #return self.getMaxConnectedNodesNum() == 1
    def getReward(self):
        # 计算当前状态的奖励的方法
        orig_edge_num = float(self.graph.num_edges[0]) + float(self.graph.num_edges[1])
        rank = self.getMaxConnectedNodesNum()
        return -float(rank) / (self.graph.max_rank * orig_edge_num)
        #return -float(self.getRemainingCNDScore()) / (orig_node_num * orig_node_num * (orig_node_num - 1) / 2)

    def getMaxConnectedNodesNum(self):
        # 获取最大连通节点数的方法
        assert self.graph
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(0,self.graph.num_nodes))
        G2.add_nodes_from(range(0,self.graph.num_nodes))

        for edge_index in self.graph.edge_sequence[0]:
            if edge_index not in self.prepare_remove_edges:
                if edge_index < self.graph.num_edges[0]:
                    G1.add_edge(self.graph.edge_sequence[1][edge_index][0],
                                self.graph.edge_sequence[1][edge_index][1])
                else:
                    G2.add_edge(self.graph.edge_sequence[1][edge_index][0],
                                self.graph.edge_sequence[1][edge_index][1])

        connected_components,rem = Mcc.MCC(G1, G2, self.remove_edge)
        rem = set()
        for idx, edge_pair in enumerate(self.graph.edge_sequence[1]):
            if edge_pair in self.remove_edge[0]:
                rem.add(idx)
            else:
                continue
        # for idx, edge_pair in enumerate(self.graph.edge_sequence[1]):
        #     if edge_pair in self.remove_edge[0] and idx < self.graph.num_edges[0]:
        #         rem.add(idx)
        #     elif edge_pair in self.remove_edge[1] and idx >= self.graph.num_edges[0]:
        #         rem.add(idx)
        #     else:
        #         continue
        # single_nodes = Mcc.find_isolated_node(connected_components)
        # single_nodes_set = {node for node_set in single_nodes for node in node_set}
        # self.single_nodes_after_act = self.single_nodes_after_act | single_nodes_set

        self.prepare_remove_edges = self.prepare_remove_edges | rem
        rank = Mcc.find_max_set_length(connected_components)
        return float(rank)

