import numpy as np
from graph import Graph
from graph_struct import GraphStruct
import torch
import Mcc
import networkx as nx
from typing import List, Tuple, Dict
#  图神经网络中批量数据的准备
'''
act_select：
    用于存储动作选择矩阵，表示在执行动作时选择的节点。对于每个样本，该矩阵的每行对应一个样本，每列对应一个节点，矩阵中的元素为1表示对应节点是执行动作时选择的节点，为0表示不是。
    
rep_global：
    用于存储全局表示矩阵，表示每个样本的全局状态。对于每个样本，该矩阵的每列对应一个样本，每行对应一个节点，矩阵中的元素为1表示对应节点是该样本的一部分，为0表示不是。

n2nsum_param：
    用于存储节点到节点的汇总参数矩阵，表示在图神经网络中节点之间的信息传递时的权重。该矩阵用于聚合节点的邻居信息。对于每个节点，该矩阵的每行对应一个节点，每列对应一个邻居节点，矩阵中的元素表示权重。

laplacian_param：
    用于存储拉普拉斯矩阵参数，表示图的拉普拉斯矩阵。对于每个节点，该矩阵的每行和每列对应一个节点，矩阵中的元素表示拉普拉斯矩阵的元素值。

subgsum_param：
    用于存储子图汇总参数矩阵，表示在图神经网络中对子图进行信息聚合时的权重。对于每个子图，该矩阵的每行对应一个子图，每列对应一个节点，矩阵中的元素表示权重。
'''
class SparseMatrix:
    def __init__(self):
        self.rowIndex = []
        self.colIndex = []
        self.value = []
        self.rowNum = 0
        self.colNum = 0

class PrepareBatchGraph:
    def __init__(self, aggregatorID):
        self.aggregatorID = aggregatorID
        self.act_select = [SparseMatrix(),SparseMatrix()]
        self.rep_global = [SparseMatrix(),SparseMatrix()]
        self.n2nsum_param = [SparseMatrix(),SparseMatrix()]
        self.laplacian_param = [SparseMatrix(),SparseMatrix()]
        self.subgsum_param = [SparseMatrix(),SparseMatrix()]
        self.idx_map_list = []
        self.subgraph_id_span = []
        self.aux_feat = []
        self.avail_act_cnt = []
        self.graph = [GraphStruct(),GraphStruct()]
        self.adj= []
        self.virtual_adj = []
        self.avail_edge = []
        self.avail_node = []
        self.isolated_nodes = []
        self.node_index_list = []

    def get_status_info(self,g: Graph,covered: List[int]):
        edge_cover = set(covered)
        remove_edge =[set(), set()]

        layout_matrix_1 = np.zeros((g.num_nodes, g.num_nodes),dtype=int)
        layout_matrix_2 = np.zeros((g.num_nodes, g.num_nodes), dtype=int)
        for edge_indx in g.edge_sequence[0]:
            if edge_indx not in edge_cover:
                if  edge_indx < g.num_edges[0]: #edge_indx小于g.num_edges证明它属于第一层图
                    layout_matrix_1[g.edge_sequence[1][edge_indx][0]]\
                    [g.edge_sequence[1][edge_indx][1]] = 1
                else: #否则就属于第二层图
                    layout_matrix_2[g.edge_sequence[1][edge_indx][0]] \
                        [g.edge_sequence[1][edge_indx][1]] = 1
        G0 = nx.from_numpy_array(layout_matrix_1)
        G1 = nx.from_numpy_array(layout_matrix_2)
        connected_components, rem= Mcc.MCC(G0, G1, remove_edge)
        isolated = Mcc.find_isolated_node(connected_components)
        isolated_nodes = [isnode for node in isolated for isnode in node]
        avail_nodes = [i for i in range(g.num_nodes) if i not in isolated_nodes]
        avail_node_sequence = [[-1 if i in isolated_nodes else 0 for i in range(g.num_nodes)],
                               [-1 if i in isolated_nodes else 0 for i in range(g.num_nodes)]]
        egde_idx_map = [-1] * (g.num_edges[0]+ g.num_edges[1])
        counter = [0, 0]
        twohop_number = [0, 0]
        avail_edge = [0, 0]
        node_twohop_counter = [{}, {}]

        rem = set()
        for idx, edge_pair in enumerate(g.edge_sequence[1]):
            if edge_pair in remove_edge[0] and idx < g.num_edges[0]:
                rem.add(idx)
            elif edge_pair in remove_edge[1] and idx >= g.num_edges[0]:
                rem.add(idx)
            else:
                continue
        edge_cover = edge_cover | rem
        for edge_idx in g.edge_sequence[0]:
            if edge_idx < g.num_edges[0]:
                layout_n = 0
            else:
                layout_n = 1

            if edge_idx  in edge_cover:
                counter[layout_n] += 1
            else:
                avail_edge[layout_n] += 1
                egde_idx_map[edge_idx] = 0


                if g.edge_sequence[1][edge_idx][0] in node_twohop_counter[layout_n]:
                    # 把有不止一个邻居的节点加起来，存在重复加行为
                    twohop_number[layout_n] += node_twohop_counter[layout_n][g.edge_sequence[1][edge_idx][0]]
                    node_twohop_counter[layout_n][g.edge_sequence[1][edge_idx][0]] = \
                        node_twohop_counter[layout_n][g.edge_sequence[1][edge_idx][0]] + 1
                else:
                    node_twohop_counter[layout_n][g.edge_sequence[1][edge_idx][0]] = 1

                if g.edge_sequence[1][edge_idx][1] in node_twohop_counter[layout_n]:
                    twohop_number[layout_n] += node_twohop_counter[layout_n][g.edge_sequence[1][edge_idx][1]]
                    node_twohop_counter[layout_n][g.edge_sequence[1][edge_idx][1]] = \
                        node_twohop_counter[layout_n][g.edge_sequence[1][edge_idx][1]] + 1
                else:
                    node_twohop_counter[layout_n][g.edge_sequence[1][edge_idx][1]] = 1
        # counter为不在动作序列里的边的数量，avail_edge是在动作序列里的连边数量
        return avail_edge, counter, twohop_number, egde_idx_map, \
               remove_edge, avail_nodes, avail_node_sequence, isolated_nodes


    def Setup_graph_input(self, idxes, g_list, covered, actions):
        self.act_select = [SparseMatrix(),SparseMatrix()]
        self.rep_global = [SparseMatrix(),SparseMatrix()]
        self.idx_map_list = []
        self.avail_act_cnt = []

        node_cnt = [0, 0]
        act_edge = [0, 0]
        for i, idx in enumerate(idxes):
            g = g_list[idx]
            temp_feat1 = []
            temp_feat2 = []
            avail_egde, counter, twohop_number, edg_idx_map,\
                remove_age, avail_node, avail_node_sequence, isolated_nodes= self.get_status_info(g, covered[i])

            temp_feat1.append(len(covered[i]) / len(g.edge_list[0]))
            temp_feat1.append(counter[0] / g.num_edges[0])
            temp_feat1.append(twohop_number[0] / (g.num_nodes * g.num_nodes))
            temp_feat1.append(1.0)
            temp_feat2.append(len(covered[i]) / len(g.edge_list[1]))
            temp_feat2.append(counter[1] / g.num_edges[1])
            temp_feat2.append(twohop_number[1] / (g.num_nodes * g.num_nodes))
            temp_feat2.append(1.0)
            temp_feat = [temp_feat1,temp_feat2]
            for j in range (2):
                node_cnt[j] += len(avail_node)
                act_edge[j] += avail_egde[j]
            self.aux_feat.append(temp_feat)
            self.idx_map_list.append(edg_idx_map)
            self.avail_edge.append(avail_egde)
            self.avail_node.append(avail_node)
            self.node_index_list.append(avail_node_sequence)
            self.isolated_nodes.append(isolated_nodes)

        for j in range(2):
            self.graph[j].resize(len(idxes), node_cnt[j])

            if actions:
                self.act_select[j].rowNum = len(idxes)
                self.act_select[j].colNum = act_edge[j] * 2
            else:
                self.rep_global[j].rowNum = node_cnt[j]
                self.rep_global[j].colNum = len(idxes)

        node_cnt = [0, 0]
        edge_cnt = [0, 0]

        for i, idx in enumerate(idxes):
            g = g_list[idx]
            edg_idx_map = self.idx_map_list[i]
            node_idx_map = self.node_index_list[i]
            # avail_node_indx_map =[self.avail_node_indx_map[i], self.avail_node_indx_map[i]]
            t = [0,0]
            for j in range(g.num_nodes):
                for h in range(2):
                    if node_idx_map[h][j] < 0:
                        continue
                    self.graph[h].add_node(i, node_cnt[h] + t[h]) #添加新节点，节点我位置在node_cnt[h] + t[h]
                    node_idx_map[h][j] = t[h]
                    if not actions:
                        self.rep_global[h].rowIndex.append(node_cnt[h] + t[h])
                        self.rep_global[h].colIndex.append(i)
                        self.rep_global[h].value.append(1.0)
                    t[h] += 1
            count = 0
            for j in range(len(edg_idx_map)):
                if edg_idx_map[j] != -1:
                    edg_idx_map[j] = count
                    count += 1
            #error
            assert t[0] == len(self.avail_node[i])

            if actions:
                act = actions[idx]
                #error

                # assert edg_idx_map[0][act] >= 0 and act >= 0 and act < len(g.edge_list[0])
                if all([num == -1 for num in edg_idx_map]):
                    continue
                else:
                    if edg_idx_map[act] >= 0:
                        if act < g.num_edges[0]:
                            lay_out = 0
                        else:
                            lay_out = 1
                        self.act_select[lay_out].rowIndex.append(i)
                        self.act_select[lay_out].colIndex.append(node_cnt[lay_out] + edg_idx_map[act])
                        self.act_select[lay_out].value.append(1.0)


            for e_inx in g.edge_sequence[0]:
                if e_inx < g.num_edges[0]:
                    lay_out = 0
                else:
                    lay_out = 1
                if edg_idx_map[e_inx] == -1:
                    continue
                if edg_idx_map[e_inx] >= 0:
                    if node_idx_map[lay_out][g.edge_sequence[1][e_inx][0]] >= 0 \
                        and node_idx_map[lay_out][g.edge_sequence[1][e_inx][1]] >=0:
                        x, y = node_idx_map[lay_out][g.edge_sequence[1][e_inx][0]] + node_cnt[lay_out], \
                               node_idx_map[lay_out][g.edge_sequence[1][e_inx][1]] + node_cnt[lay_out]
                        self.graph[lay_out].add_edge(edge_cnt[lay_out], x, y)
                        edge_cnt[lay_out] += 1
                        self.graph[lay_out].add_edge(edge_cnt[lay_out], y, x)
                        edge_cnt[lay_out] += 1

            node_cnt[0] += len(self.avail_node[i])
            node_cnt[1] += len(self.avail_node[i])


        #error
        assert node_cnt[0] == self.graph[0].num_nodes
        result_list = self.n2n_construct(self.aggregatorID)
        self.n2nsum_param = result_list[0]
        self.laplacian_param = result_list[1]
        self.adj = result_list[2]    #邻接矩阵
        result_list1 = self.subg_construct()
        self.subgsum_param = result_list1[0]
        self.virtual_adj = result_list1[1]

        for j in range(2):
            self.act_select[j] = self.convert_sparse_to_tensor(self.act_select[j])
            self.rep_global[j] = self.convert_sparse_to_tensor(self.rep_global[j])
            self.n2nsum_param[j] = self.convert_sparse_to_tensor(self.n2nsum_param[j])
            self.laplacian_param[j] = self.convert_sparse_to_tensor(self.laplacian_param[j])
            self.subgsum_param[j] = self.convert_sparse_to_tensor(self.subgsum_param[j])
    '''
    act_select：用于存储动作选择矩阵，表示在执行动作时选择的节点。对于每个样本，该矩阵的每行对应一个样本，每列对应一个节点，矩阵中的元素为1表示对应节点是执行动作时选择的节点，为0表示不是。
    rep_global：用于存储全局表示矩阵，表示每个样本的全局状态。对于每个样本，该矩阵的每列对应一个样本，每行对应一个节点，矩阵中的元素为1表示对应节点是该样本的一部分，为0表示不是。
    n2nsum_param：用于存储节点到节点的汇总参数矩阵，表示在图神经网络中节点之间的信息传递时的权重。该矩阵用于聚合节点的邻居信息。对于每个节点，该矩阵的每行对应一个节点，每列对应一个邻居节点，矩阵中的元素表示权重。
    laplacian_param：用于存储拉普拉斯矩阵参数，表示图的拉普拉斯矩阵。对于每个节点，该矩阵的每行和每列对应一个节点，矩阵中的元素表示拉普拉斯矩阵的元素值。
    subgsum_param：用于存储子图汇总参数矩阵，表示在图神经网络中对子图进行信息聚合时的权重。对于每个子图，该矩阵的每行对应一个子图，每列对应一个节点，矩阵中的元素表示权重。
    '''

    def SetupTrain(self, idxes, g_list, covered, actions):
        self.Setup_graph_input(idxes, g_list, covered, actions)

    def SetupPredAll(self, idxes, g_list, covered):
        self.Setup_graph_input(idxes, g_list, covered, None)
    '''
    def convert_sparse_to_tensor(self, matrix):
        indices = np.column_stack((matrix.rowIndex, matrix.colIndex))
        return torch.sparse.FloatTensor(torch.LongTensor(indices).t(), torch.FloatTensor(matrix.value),
                                         torch.Size([matrix.rowNum, matrix.colNum]))
    '''

    def convert_sparse_to_tensor(self, matrix):
        rowIndex= matrix.rowIndex
        colIndex= matrix.colIndex
        data= matrix.value
        rowNum= matrix.rowNum
        colNum= matrix.colNum
        indices = np.mat([rowIndex, colIndex]).transpose()

        index = torch.tensor(np.transpose(np.array(indices)))
        value = torch.Tensor(np.array(data))
        #index, value = torch_sparse.coalesce(index, value, m=rowNum, n=colNum)
        return_dict = {"index": index, "value": value, "m":rowNum, "n":colNum}
        return return_dict

    '''
    def graph_resize(self, size, node_cnt):
        self.graph = Graph(size, node_cnt)

    def graph_add_node(self, i, node):
        self.graph.add_node(i, node)

    def graph_add_edge(self, edge, x, y):
        self.graph.add_edge(edge, x, y)
    '''
    #这段代码讲得的是将已有的图结构传入 result和result_laplacian
    #然后result里每个边的值是1，result_laplacian应该对应的是节点，每个节点的值要根据它的邻居来计算
    #adj_matrixs 临界矩阵
    def n2n_construct(self, aggregatorID):
        result = [SparseMatrix(),SparseMatrix()]  #用来存储图的稀疏矩阵
        result_laplacian = [SparseMatrix(),SparseMatrix()] #用来存储拉普拉斯矩阵
        adj_matrixs = [] #用来存储邻接矩阵
        for h in range(2):
            result[h].rowNum = self.graph[h].num_nodes
            result[h].colNum = self.graph[h].num_nodes
            result_laplacian[h].rowNum = self.graph[h].num_nodes
            result_laplacian[h].colNum = self.graph[h].num_nodes

            for i in range(self.graph[h].num_nodes):
                list1 = self.graph[h].in_edges.head[i]

                if len(list1) > 0:
                    result_laplacian[h].value.append(len(list1))
                    result_laplacian[h].rowIndex.append(i)
                    result_laplacian[h].colIndex.append(i)

                for j in range(len(list1)):
                    if aggregatorID == 0:
                        result[h].value.append(1.0)
                    elif aggregatorID == 1:
                        result[h].value.append(1.0 / len(list1))
                    elif aggregatorID == 2:
                        #neighborDegree = len(self.graph.in_edges.head[list1[j].second])
                        neighborDegree = len(self.graph[h].in_edges.head[list1[j][1]])
                        selfDegree = len(list1)
                        norm = np.sqrt(neighborDegree + 1) * np.sqrt(selfDegree + 1)
                        result[h].value.append(1.0 / norm)

                    result[h].rowIndex.append(i)
                    #result[i].colIndex.append(list1[j].second)
                    result[h].colIndex.append(list1[j][1])
                    result_laplacian[h].value.append(-1.0)
                    result_laplacian[h].rowIndex.append(i)
                    #result[i].result_laplacian[i].colIndex.append(list1[j].second)
                    result_laplacian[h].colIndex.append(list1[j][1])

            adj_matrix = np.zeros((self.graph[h].num_nodes,self.graph[h].num_nodes))
            for edge in self.graph[h].edge_list:
                i,j=edge
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
            adj_matrixs.append(adj_matrix)
        return [result,result_laplacian,adj_matrixs]

    '''
    def e2n_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_nodes
        result.colNum = self.graph.num_edges

        for i in range(self.graph.num_nodes):
            list1 = self.graph.in_edges.head[i]
            for j in range(len(list1)):
                result.value.append(1.0)
                result.rowIndex.append(i)
                result.colIndex.append(list1[j].first)
        return result

    def n2e_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_edges
        result.colNum = self.graph.num_nodes

        for i in range(self.graph.num_edges):
            result.value.append(1.0)
            result.rowIndex.append(i)
            result.colIndex.append(self.graph.edge_list[i].first)

        return result

    def e2e_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_edges
        result.colNum = self.graph.num_edges

        for i in range(self.graph.num_edges):
            node_from, node_to = self.graph.edge_list[i]
            list1 = self.graph.in_edges.head[node_from]

            for j in range(len(list1)):
                if list1[j].second == node_to:
                    continue
                result.value.append(1.0)
                result.rowIndex.append(i)
                result.colIndex.append(list1[j].first)

        return result
    '''

    def subg_construct(self):
        result = [SparseMatrix(),SparseMatrix()]
        virtual_adjs = []

        for h in range(2):
            result[h].rowNum = self.graph[h].num_subgraph
            result[h].colNum = self.graph[h].num_nodes

            subgraph_id_span = []
            start = 0
            end = 0

            for i in range(self.graph[h].num_subgraph):
                #self.graph[h].subgraph.head[i] 这里是现有的节点
                list1 = self.graph[h].subgraph.head[i]
                end = start + len(list1) - 1

                for j in range(len(list1)):
                    result[h].value.append(1.0)
                    result[h].rowIndex.append(i)
                    result[h].colIndex.append(list1[j])

                if len(list1) > 0:
                    subgraph_id_span.append((start, end))
                else:
                    subgraph_id_span.append((self.graph[h].num_nodes, self.graph[h].num_nodes))
                start = end + 1
            virtual_adj = np.zeros((result[h].rowNum,result[h].colNum))
            for i in range(len(result[h].value)):
                row_idx = result[h].rowIndex[i]
                col_idx = result[h].colIndex[i]
                weight = result[h].value[i]
                virtual_adj[row_idx][col_idx] = weight
            virtual_adjs.append(virtual_adj)
        return [result,virtual_adjs]


