# utils.py

from typing import List
from disjoint_set import DisjointSet
from graphutils import GraphUtil
import networkx as nx
import Mcc
class Utils:
    def __init__(self):
        self.MaxWccSzList = []

    #对解决方案solution按照指定的策略重新排序
    def reInsert(self, graph, solution, allVex, decreaseStrategyID, reinsertEachStep):

        return self.reInsert_inner(solution, graph, allVex, decreaseStrategyID, reinsertEachStep)

    def reInsert_inner(self, beforeOutput, graph, allVex, decreaseStrategyID, reinsertEachStep):
        currentAdjListGraph = []
        backupCompletedAdjListGraph = graph.adj_list.copy()
        currentAllVex = [False] * graph.num_nodes

        for eachV in allVex:
            currentAllVex[eachV] = True

        leftOutput = set(beforeOutput)
        finalOutput = []
        disjoint_Set = DisjointSet(graph.num_nodes)
        graphutil = GraphUtil()

        while leftOutput:
            batchList = []
            for eachNode in leftOutput:
                decreaseValue = decreaseStrategyID.decreaseComponentNumIfAddNode(
                    backupCompletedAdjListGraph, currentAllVex, disjoint_Set, eachNode
                )
                batchList.append((decreaseValue, eachNode))

            batchList.sort()

            for i in range(min(reinsertEachStep, len(batchList))):
                finalOutput.append(batchList[i][1])
                leftOutput.remove(batchList[i][1])
                graphutil.recover_add_node(
                    backupCompletedAdjListGraph,
                    currentAllVex,
                    currentAdjListGraph,
                    batchList[i][1],
                    disjoint_Set,
                )

        finalOutput.reverse()
        return finalOutput

    # 计算给定解决方案solution在图graph上的鲁棒性。通过逐步添加节点，计算每个步骤下最大连通分量的大小，
    def getRobustness(self, graph, solution):
        assert(graph)
        self.MaxWccSzList = [] # 用于存储每个节点加入后的最大连通子集大小（权重）的列表。
        backupCompletedAdjListGraph = graph.adj_list.copy()   #复制了原始图的邻接列表
        currentAdjList = [[],[]]  #当前的邻接列表，用于存储当前考虑的节点及其邻居
        graphutil = GraphUtil()
        disjoint_Set1 = DisjointSet(graph.num_nodes) #用于处理不相交集合的操作
        disjoint_Set2 = DisjointSet(graph.num_nodes)  # 用于处理不相交集合的操作
        disjoint_Set = [disjoint_Set1,disjoint_Set2]
        backupAllVex = [False] * graph.num_nodes #一个与图中节点数量相等的列表，初始化为False，可能用于标记节点是否已被考虑
        totalMaxNum = 0.0 #累加考虑节点后的最大连通子集大小（权重）
        temp = 0.0 #暂存每次迭代中的最大连通子集大小
        covered_nodes = []
        for Node in reversed(solution):
            graphutil.recover_add_node(
                backupCompletedAdjListGraph, backupAllVex, currentAdjList, Node, disjoint_Set
            )
            covered_nodes.append(Node)
            G1 = nx.Graph()
            G2 = nx.Graph()
            G1.add_nodes_from(range(0, graph.num_nodes))
            G2.add_nodes_from(range(0, graph.num_nodes))
            for i in covered_nodes:
                for j in graph.adj_list[0][i]:
                    #for k in range(len(currentAdjList[0])):
                    if len(currentAdjList[0]) >= i+1:
                        if j in currentAdjList[0][i]:
                            G1.add_edge(i, j)
            for i in covered_nodes:
                for j in graph.adj_list[1][i]:
                    #for k in range(len(currentAdjList[1])):
                    if len(currentAdjList[1]) >= i+1:
                        if j in currentAdjList[1][i]:
                            G2.add_edge(i, j)
            #print(G1.edges)
            remove_edge = [set(),set()]
            connected_components, remo= Mcc.MCC(G1, G2, remove_edge)
            rank = Mcc.find_max_set_length(connected_components)
            #print(rank)
            totalMaxNum += rank
            self.MaxWccSzList.append(rank/graph.num_nodes)
            temp = rank
        totalMaxNum = totalMaxNum - temp
        self.MaxWccSzList.reverse()
        return totalMaxNum / (graph.num_nodes*graph.num_nodes)

    #图graph的最大弱连通分量的大小。使用了并查集（DisjointSet）来合并连通分量
    def getMxWccSz(self, graph):
        disjoint_Set = DisjointSet(graph.num_nodes)
        for i in range(len(graph.adj_list)):
            for j in range(len(graph.adj_list[i])):
                disjoint_Set.merge(i, graph.adj_list[i][j])
        return disjoint_Set.max_rank_count

    #计算图graph中所有节点的介数中心性。通过广度优先搜索计算最短路径和计算介数中心性
    def Betweenness(self, graph):
        nvertices = graph.num_nodes
        CB = [0.0] * nvertices
        norm = (nvertices - 1) * (nvertices - 2)

        for i in range(nvertices):
            PredList = [[] for _ in range(nvertices)]
            d = [4294967295] * nvertices
            d[i] = 0
            sigma = [0] * nvertices
            sigma[i] = 1
            delta = [0.0] * nvertices
            Q = []
            S = []

            Q.append(i)

            while Q:
                u = Q.pop(0)
                S.append(u)

                for v in graph.adj_list[u]:
                    if d[v] == 4294967295:
                        d[v] = d[u] + 1
                        Q.append(v)

                    if d[v] == d[u] + 1:
                        sigma[v] += sigma[u]
                        PredList[v].append(u)

            while S:
                u = S.pop()
                for j in PredList[u]:
                    delta[j] += (sigma[j] / sigma[u]) * (1 + delta[u])

                if u != i:
                    CB[u] += delta[u]

            PredList = []
            d = []
            sigma = []
            delta = []

        for i in range(nvertices):
            CB[i] = CB[i] / norm
        print(CB)
        return CB
