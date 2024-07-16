import numpy as np
from GMM import GMM
import sys
from collections import defaultdict
import torch
import networkx as nx
import Mcc

class Graph:
    def __init__(self, N = 0):
        # N初始化节点数量
        self.num_nodes = N
        self.adj_list = []
        self.edge_list = []
        self.num_edges = []
        self.max_rank = 0
        self.max_edge_number =[]
        self.edge_sequence = [[], []]
        num_nodes_layer1 = N
        num_nodes_layer2 = N
        if N != 0:
            link1,link2 = GMM(N, 0.5)
            num_edges_layer1 = len(link1)
            num_edges_layer2 = len(link2)
            edges_from_layer1 = [item[0] for item in link1]
            edges_to_layer1 = [item[1] for item in link1]
            edges_from_layer2 = [item[0] for item in link2]
            edges_to_layer2 = [item[1] for item in link2]
            # 创建第一层图
            self.reshape_graph(num_nodes_layer1, num_edges_layer1, edges_from_layer1, edges_to_layer1)
            for i in range(num_edges_layer1):
                self.edge_sequence[0].append(i)
            for i in range(num_edges_layer1):
                self.edge_sequence[1].append((edges_from_layer1[i], edges_to_layer1[i]))
            # 创建第二层图
            self.reshape_graph(num_nodes_layer2, num_edges_layer2, edges_from_layer2, edges_to_layer2)
            for i in range(num_edges_layer1, (num_edges_layer2 + num_edges_layer1)):
                self.edge_sequence[0].append(i)
            for i in range(num_edges_layer2):
                self.edge_sequence[1].append((edges_from_layer2[i], edges_to_layer2[i]))

            self.ori_rank()

    def reshape_graph(self, num_nodes, num_edges, edges_from, edges_to):
        self.num_nodes = num_nodes
        self.num_edges.append(num_edges)
        adj_list = [[] for _ in range(num_nodes)]
        edge_list = [(edges_from[i], edges_to[i]) for i in range(num_edges)]
        for i in range(num_edges):
            x, y = edges_from[i], edges_to[i]
            adj_list[x].append(y)
            adj_list[y].append(x)
        self.adj_list.append(adj_list)
        self.edge_list.append(edge_list)


    def ori_rank(self):
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(0,self.num_nodes))
        G2.add_nodes_from(range(0,self.num_nodes))
        for i in range(0,self.num_nodes):
            for j in self.adj_list[0][i]:
                    G1.add_edge(i,j)
        for i in range(0,self.num_nodes):
            for j in self.adj_list[1][i]:
                    G2.add_edge(i,j)
        remove_edge = [set(),set()]
        connected_components, rem = Mcc.MCC(G1,G2,remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)
        self.max_edge_number = Mcc.find_max_edge_number(connected_components)



class GSet:
    def __init__(self):
        self.graph_pool = {}

    def InsertGraph(self, gid, graph):
        assert gid not in self.graph_pool
        self.graph_pool[gid] = graph

    def Sample(self):
        assert self.graph_pool
        gid = np.random.choice(list(self.graph_pool.keys()))
        return self.graph_pool[gid]

    def Get(self, gid):
        assert gid in self.graph_pool
        return self.graph_pool[gid]

    def Clear(self):
        self.graph_pool.clear()


class Graph_ACM:
    def __init__(self, adj1, adj2):
        # N初始化节点数量
        self.num_nodes = adj1.shape[0]
        self.adj_list = []
        self.edge_list = []
        self.num_edges = []
        edges1 = np.transpose(np.triu(adj1).nonzero())
        edges2 = np.transpose(np.triu(adj2).nonzero())
        self.edge_list = [edges1, edges2]
        self.num_edges = [len(edges1), len(edges2)]
        self.max_rank = 0  #最大联通集团个数
        adj_lists_temp = defaultdict(set)
        [row, col] = np.where(adj1 == 1)
        for i in range(row.size):
            adj_lists_temp[row[i]].add(col[i])
            adj_lists_temp[col[i]].add(row[i])
        self.adj_list.append(adj_lists_temp)

        adj_lists_temp = defaultdict(set)
        [row, col] = np.where(adj2 == 1)
        for i in range(row.size):
            adj_lists_temp[row[i]].add(col[i])
            adj_lists_temp[col[i]].add(row[i])
        self.adj_list.append(adj_lists_temp)
        self.ori_rank()

    def ori_rank(self):
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(0, self.num_nodes))
        G2.add_nodes_from(range(0, self.num_nodes))
        for i in range(0, self.num_nodes):
            for j in self.adj_list[0][i]:
                G1.add_edge(i, j)
        for i in range(0, self.num_nodes):
            for j in self.adj_list[1][i]:
                G2.add_edge(i, j)
        remove_edge = [set(), set()]
        connected_components = Mcc.MCC(G1, G2, remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)


class Graph_test:
    def __init__(self, G1, G2):
        # N初始化节点数量
        self.num_nodes = len(G1.nodes)
        self.adj_list = [list(G1.adjacency()), list(G2.adjacency())]
        self.edge_list = [G1.edges(), G2.edges()]
        self.num_edges = [len(self.edge_list[0]), len(self.edge_list[1])]
        self.max_rank = 0
        self.weights = [{}, {}]

        self.edge_sequence = [[], []]

        for i in range(G1.number_of_edges()):
            self.edge_sequence[0].append(i)
        for e1, e2 in G1.edges:
            self.edge_sequence[1].append((e1, e2))
        for i in range(G1.number_of_edges(), (G1.number_of_edges()+G2.number_of_edges())):
            self.edge_sequence[0].append(i)
        for e1, e2 in G2.edges:
            self.edge_sequence[1].append((e1, e2))


        self.ori_rank(G1, G2)


    def ori_rank(self, G1, G2):
        remove_edge = [set(), set()]
        connected_components, rem = Mcc.MCC(G1.copy(), G2.copy(), remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)
        self.max_edge_number = Mcc.find_max_edge_number(connected_components)
        return G1, G2