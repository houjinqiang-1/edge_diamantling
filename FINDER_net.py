from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse
import numpy as np
from MRGNN.encoders import Encoder
from MRGNN.aggregators import MeanAggregator, LSTMAggregator, PoolAggregator
from MRGNN.utils import LogisticRegression
from MRGNN.mutil_layer_weight import LayerNodeAttention_weight, Cosine_similarity, SemanticAttention, \
    BitwiseMultipyLogis


# cudnn.benchmark = False
# cudnn.deterministic = True
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# out = torch_sparse.spmm(index, value, m, n, matrix)
class FINDER_net(nn.Module):
    def __init__(self, SupervisedGraphSage, Encoder1, Encoder2, Aggregator, layerNodeAttention_weight,
                 embedding_size=64, w_initialization_std=1, reg_hidden=64, max_bp_iter=3,
                 embeddingMethod=1, aux_dim=4, device=None, node_attr=False):
        super(FINDER_net, self).__init__()

        self.SupervisedGraphSage = SupervisedGraphSage
        self.Encoder1 = Encoder1
        self.Encoder2 = Encoder2
        self.Aggregator = Aggregator
        self.layerNodeAttention_weight = layerNodeAttention_weight
        # self.rand_generator = torch.normal
        # see https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
        self.rand_generator = lambda mean, std, size: torch.fmod(torch.normal(mean, std, size=size), 2)
        self.embedding_size = embedding_size
        self.w_initialization_std = w_initialization_std
        self.reg_hidden = reg_hidden
        self.max_bp_iter = max_bp_iter
        self.embeddingMethod = embeddingMethod
        self.aux_dim = aux_dim
        self.device = device
        self.node_attr = node_attr
        self.act = nn.ReLU()

        # 初始化神经网络权重矩阵
        # [2, embed_dim]
        self.w_n2l = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(2, self.embedding_size)))
        #初始化神经网络权重矩阵
        # [embed_dim, embed_dim]
        self.p_node_conv = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,
                                                                           size=(self.embedding_size, self.embedding_size)))
        # ##以上两个初始化只是维度的不同
        # self.p_node_conv2 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,
        #                                                                     size=(self.embedding_size,
        #                                                                           self.embedding_size)))
        # # [2*embed_dim, embed_dim]
        # self.p_node_conv3 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,
        #                                                                     size=(2 * self.embedding_size,
        #                                                                           self.embedding_size)))

        # [reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #嵌入层到隐藏层的映射
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(
                                                                                 self.embedding_size, self.reg_hidden)))

            # [reg_hidden+aux_dim, 1]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            #隐藏层到输出层的映射
            self.h2_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(self.reg_hidden + self.aux_dim, 1)))
            # [reg_hidden2 + aux_dim, 1]
            self.last_w = self.h2_weight
        else:
            # [2*embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,
                                                                             size=(
                                                                                 2 * self.embedding_size,
                                                                                 self.reg_hidden)))
            # [2*embed_dim, reg_hidden]
            self.last_w = self.h1_weight
        # self.w_layer1 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,
        #                                                                 size=(embedding_size, 128)))
        # self.w_layer2 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,
        #                                                                 size=(128, 1)))
        #
        # self.flag = 0

        ## [embed_dim, 1]
        # cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)
        self.cross_product = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(self.embedding_size, 1)))

    def train_forward(self, node_input, subgsum_param, n2nsum_param, action_select, aux_input, adj, v_adj, input_graph):
        
        nodes_cnt = n2nsum_param[0]['m']
        if (self.node_attr == False):
            # [node_cnt, 2]
            node_input = torch.ones((nodes_cnt, 2)).to(self.device)
                                 
        y_nodes_size = subgsum_param[0]['m']
        y_node_input = torch.ones((y_nodes_size,2)).to(self.device)  #初始化 （1行，2列）全1矩阵
        
        input_message = torch.matmul(node_input, self.w_n2l)
        #[node_cnt, embed_dim]  # no sparse
        #input_potential_layer = tf.nn.relu(input_message)
        input_potential_layer = self.act(input_message)

        # # no sparse
        # [batch_size, embed_dim]
        #y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        y_input_message = torch.matmul(y_node_input, self.w_n2l)
        #[batch_size, embed_dim]  # no sparse
        #y_input_potential_layer = tf.nn.relu(y_input_message)
        y_input_potential_layer = self.act(y_input_message)

        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
        cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
        
        n2npool0 = torch_sparse.spmm(n2nsum_param[0]['index'], n2nsum_param[0]['value'],\
                n2nsum_param[0]['m'], n2nsum_param[0]['n'], cur_message_layer)
        #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #node_linear = tf.matmul(n2npool, p_node_conv)
        node_linear0 = torch.matmul(n2npool0, self.p_node_conv)

        n2npool1 = torch_sparse.spmm(n2nsum_param[1]['index'], n2nsum_param[1]['value'],\
                n2nsum_param[1]['m'], n2nsum_param[1]['n'], cur_message_layer)
        #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #node_linear = tf.matmul(n2npool, p_node_conv)
        node_linear1 = torch.matmul(n2npool1, self.p_node_conv)
        # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        
        #OLD y_n2npool = torch.matmul(subgsum_param, cur_message_layer)
        y_n2npool0 = torch_sparse.spmm(subgsum_param[0]['index'], subgsum_param[0]['value'],\
                subgsum_param[0]['m'], subgsum_param[0]['n'], cur_message_layer)

        #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        y_node_linear0 = torch.matmul(y_n2npool0, self.p_node_conv)
        
        y_n2npool1 = torch_sparse.spmm(subgsum_param[1]['index'], subgsum_param[1]['value'],\
                subgsum_param[1]['m'], subgsum_param[1]['n'], cur_message_layer)

        #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        y_node_linear1 = torch.matmul(y_n2npool1, self.p_node_conv)
        
        node_input = torch.stack((node_linear0,node_linear1),axis=0)
        y_node_input = torch.stack((y_node_linear0,y_node_linear1),axis=0)
    
        node_input = torch.cat((node_input, y_node_input), axis=1)
        
        adj = np.array(adj)
        v_adj = np.array(v_adj)
        adj = np.concatenate((adj, np.zeros((adj.shape[0], adj.shape[1], y_nodes_size))), axis=2)
        v_adj = np.concatenate((v_adj, np.zeros((v_adj.shape[0], v_adj.shape[1], y_nodes_size))), axis=2)
        adj = np.concatenate((adj, v_adj), axis=1)
        if self.embeddingMethod == 1:  # MRGNN
            num_nodes = adj.shape[1]
            emb_dim = self.embedding_size
            adj_lists = []
            lay_num = 2
            for ki in range(lay_num):
                adj_lists_temp = defaultdict(set)
                [row, col] = np.where(adj[ki] == 1)
                for i in range(row.size):
                    adj_lists_temp[row[i]].add(col[i])
                    # adj_lists_temp[col[i]].add(row[i])
                adj_lists.append(adj_lists_temp)

            #features = []
            # for kik in range(lay_num):
            #     features_temp = nn.Embedding(num_nodes, node_input[kik].shape[1])
            #     features_temp.weigh t = nn.Parameter(torch.FloatTensor(node_input[kik].cpu()), requires_grad=True)
            #     features_temp.cuda(self.device)
            #     features.append(features_temp)
            nodes = np.array(list(range(nodes_cnt + y_nodes_size)))

            message_layer = self.SupervisedGraphSage(nodes, node_input, adj_lists,
                                                     self.Encoder1, self.Encoder2, self.layerNodeAttention_weight,
                                                     self.Aggregator)
            cur_message_layer = message_layer[:, :nodes_cnt, :]


                # 获取当前层的消息张量

            # for l in range(lay_num):
            #     temp_lenth = 0
            #     for i in range(len(input_graph[0].in_edges.head)):
            #         edge_pair_1 = i
            #         for j in  input_graph[l].in_edges.head[i]:
            #             edge_pair_2 = j[1]
            #         req_re = torch.mul(cur_message_layer[l][edge_pair_1],
            #                            cur_message_layer[l][edge_pair_2])
            #         edge_curr_message_layer[l].append(req_re)
            #
            #         pass

            # for j in range(0, len(input_graph[l].edge_list), 2):
            #     edge_pair_1 = input_graph[l].edge_list[j][0]
            #     edge_pair_2 = input_graph[l].edge_list[j][1]
            #     req_re = torch.mul(rep_aux[edge_pair_1], rep_aux[edge_pair_2])
            #     req_result[l].append(req_re)
            # req_result[l] = torch.stack(req_result[l])
            y_cur_message_layer = message_layer[:, nodes_cnt:, :]
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)

        q = [[], []]
        q = 0
        for l in range(lay_num):
            edge_curr_message_layer = []
            node_embedding = cur_message_layer[l]
            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
            y_potential = y_cur_message_layer[l].cuda(self.device)
            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            # action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)
            # OLD action_embed = torch.matmul(action_select, cur_message_layer)
            cur_messages = cur_message_layer[l]
            # 获取所有节点的入边
            in_edges = input_graph[l].in_edges.head
            # 获取所有入边对应的头尾节点索引
            edge_pairs = [(i, j[1]) for i, edges in enumerate(in_edges) for j in edges]
            # 将入边的头尾节点索引转换为张量
            edge_pairs = torch.tensor(edge_pairs)
            # 从当前消息张量中按索引取出对应的消息，并逐元素相乘
            edge_message = cur_messages[edge_pairs[:, 0]] * cur_messages[edge_pairs[:, 1]]
            # 将结果添加到对应的列表中


            action_embed = torch_sparse.spmm(action_select[l]['index'], action_select[l]['value'], \
                                             action_select[l]['m'], action_select[l]['n'],
                                             edge_message.cuda(self.device))

            # # [batch_size, embed_dim, embed_dim]
            # temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1))
            temp = torch.matmul(torch.unsqueeze(action_embed, dim=2), torch.unsqueeze(y_potential, dim=1))
            # [batch_size, embed_dim]
            # Shape = tf.shape(action_embed)
            Shape = action_embed.size()
            # [batch_size, embed_dim], first transform
            # embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)
            embed_s_a = torch.reshape(torch.matmul(temp, torch.reshape(torch.tile(self.cross_product, [Shape[0], 1]), \
                                                                       [Shape[0], Shape[1], 1])), Shape)

            # [batch_size, 2 * embed_dim]
            last_output = embed_s_a



            if self.reg_hidden > 0:
                # [batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
                # hidden = tf.matmul(embed_s_a, h1_weight)

                hidden = torch.matmul(embed_s_a, self.h1_weight)
                # [batch_size, reg_hidden]
                # last_output = tf.nn.relu(hidden)
                last_output = self.act(hidden)

            # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
            # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
            # last_output = tf.concat([last_output, self.aux_input], 1)
            last_output = torch.concat([last_output, aux_input[:,l,:]], 1)
            # if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
            # if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
            # q_pred = tf.matmul(last_output, last_w)
            q_pred = torch.matmul(last_output, self.last_w)
            q += q_pred
        return q, cur_message_layer.cuda(self.device)

    def test_forward(self, node_input, subgsum_param, n2nsum_param, rep_global, aux_input, adj, v_adj,
                     input_graph):

        nodes_cnt = n2nsum_param[0]['m']  #路径数量
        if (self.node_attr == False):     #
            # [node_cnt, 2] #36*2的tensor张量 初始化全一矩阵
            node_input = torch.ones((nodes_cnt, 2)).to(self.device)
        #subgsum_param用于存储子图汇总参数矩阵，表示在图神经网络中对子图进行信息聚合时的权重。
        # 对于每个子图，该矩阵的每行对应一个子图，每列对应一个节点，矩阵中的元素表示权重。
        #subgsum_param[0]['m'] 表示子图个数
        y_nodes_size = subgsum_param[0]['m']
        y_node_input = torch.ones((y_nodes_size,2)).to(self.device) #初始化全1矩阵
        
        input_message = torch.matmul(node_input, self.w_n2l) #矩阵乘法
        #[node_cnt, embed_dim]  # no sparse
        #input_potential_layer = tf.nn.relu(input_message)
        input_potential_layer = self.act(input_message) #将矩阵乘积结果输入激活函数 act 中，得到激活后的输出。
        # # no sparse
        # [batch_size, embed_dim]
        #y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        #矩阵乘法
        y_input_message = torch.matmul(y_node_input, self.w_n2l)
        #[batch_size, embed_dim]  # no sparse
        #y_input_potential_layer = tf.nn.relu(y_input_message)
        y_input_potential_layer = self.act(y_input_message)

        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
        cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)#归一化操作

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)

        #cur_message_layer是网络图信息，y_cur_message_layer是虚拟节点信息
        #n2nsum_param[0]['index']指的是位置的索引，n2nsum_param[0]['value']指的是位置的权重
        #n2nsum_param[0]['m'], n2nsum_param[0]['n']矩阵的行数和列数
        #torch_sparse.spmm进行稀疏矩阵乘法
        n2npool0 = torch_sparse.spmm(n2nsum_param[0]['index'], n2nsum_param[0]['value'],\
                n2nsum_param[0]['m'], n2nsum_param[0]['n'], cur_message_layer)
        #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #node_linear = tf.matmul(n2npool, p_node_conv)
        node_linear0 = torch.matmul(n2npool0, self.p_node_conv)

        n2npool1 = torch_sparse.spmm(n2nsum_param[1]['index'], n2nsum_param[1]['value'],\
                n2nsum_param[1]['m'], n2nsum_param[1]['n'], cur_message_layer)
        #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #node_linear = tf.matmul(n2npool, p_node_conv)
        node_linear1 = torch.matmul(n2npool1, self.p_node_conv)
        # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        
        #OLD y_n2npool = torch.matmul(subgsum_param, cur_message_layer)
        y_n2npool0 = torch_sparse.spmm(subgsum_param[0]['index'], subgsum_param[0]['value'],\
                subgsum_param[0]['m'], subgsum_param[0]['n'], cur_message_layer)

        #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        y_node_linear0 = torch.matmul(y_n2npool0, self.p_node_conv)
        
        y_n2npool1 = torch_sparse.spmm(subgsum_param[1]['index'], subgsum_param[1]['value'],\
                subgsum_param[1]['m'], subgsum_param[1]['n'], cur_message_layer)

        #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        y_node_linear1 = torch.matmul(y_n2npool1, self.p_node_conv)

        #torch.stack我们想要将这两个张量沿着第0维度（即新的维度0）堆叠起来，形成一个新的张量
        #将两层的嵌入向量堆叠起来
        node_input = torch.stack((node_linear0,node_linear1),axis=0)
        y_node_input = torch.stack((y_node_linear0,y_node_linear1),axis=0)
        node_input = torch.cat((node_input, y_node_input), axis=1)
        ######### 调试
        adj = np.array(adj) #邻接矩阵
        v_adj = np.array(v_adj)
        #########
        adj = np.concatenate((adj, np.zeros((adj.shape[0], adj.shape[1], y_nodes_size))), axis=2)
        v_adj = np.concatenate((v_adj, np.zeros((v_adj.shape[0], v_adj.shape[1], y_nodes_size))), axis=2)
        adj = np.concatenate((adj, v_adj), axis=1)
        if self.embeddingMethod == 1:  # MRGNN
            num_nodes = adj.shape[1]
            emb_dim = self.embedding_size
            adj_lists = []
            lay_num = 2
            for ki in range(lay_num):
                adj_lists_temp = defaultdict(set)
                [row, col] = np.where(adj[ki] == 1)
                for i in range(row.size):
                    adj_lists_temp[row[i]].add(col[i])
                    # adj_lists_temp[col[i]].add(row[i])
                adj_lists.append(adj_lists_temp)

            # features = []
            # for kik in range(lay_num):
            #     features_temp = nn.Embedding(num_nodes, node_input[kik].shape[1])
            #     features_temp.weight = nn.Parameter(torch.FloatTensor(node_input[kik].cpu()), requires_grad=True)
            #     features_temp.cuda(self.device)
            #     features.append(features_temp)

            nodes = np.array(list(range(nodes_cnt + y_nodes_size)))
            message_layer = self.SupervisedGraphSage(nodes, node_input, adj_lists,
                                                     self.Encoder1, self.Encoder2, self.layerNodeAttention_weight,
                                                     self.Aggregator)
            cur_message_layer = message_layer[:, :nodes_cnt, :]
            y_cur_message_layer = message_layer[:, nodes_cnt:, :]
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)

            # cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            # y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)

        q = [[], []]
        edge_result = [[], []]
        req_result = [[], []]
        for l in range(lay_num):
            y_potential = y_cur_message_layer[l].cuda(self.device)
            # [node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
            # OLD rep_y = torch.matmul(rep_global, y_potential)
            rep_y = torch_sparse.spmm(rep_global[l]['index'], rep_global[l]['value'].cuda(self.device), \
                                      rep_global[l]['m'], rep_global[l]['n'], y_potential.cuda(self.device))
            # [[node_cnt, embed_dim], [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim]
            # embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)
            # # [node_cnt, embed_dim, embed_dim]
            # torch.unsqueeze() 它的作用是在指定的维度上对张量进行维度扩展
            temp1 = torch.matmul(torch.unsqueeze(cur_message_layer[l].cuda(self.device), dim=2),
                                 torch.unsqueeze(rep_y, dim=1))
            # [node_cnt embed_dim]
            Shape1 = cur_message_layer[l].size()
            # [batch_size, embed_dim], first transform
            #torch.tile() 函数用于在指定维度上复制张量的内容
            #torch.reshape()将张量转化为指定形状
            embed_s_a_all = torch.reshape(torch.matmul(temp1,
                                                       torch.reshape(torch.tile(self.cross_product, [Shape1[0], 1]),
                                                                     [Shape1[0], Shape1[1], 1])), Shape1)

            # [node_cnt, 2 * embed_dim]
            last_output = embed_s_a_all

            for j in range(0, len(input_graph[l].edge_list), 2):
                edge_pair_1 = input_graph[l].edge_list[j][0]
                edge_pair_2 = input_graph[l].edge_list[j][1]
                result = torch.mul(last_output[edge_pair_1], last_output[edge_pair_2])
                edge_result[l].append(result)
            edge_result[l] = torch.stack(edge_result[l])


            if self.reg_hidden > 0:
                # [node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
                hidden = torch.matmul(edge_result[l], self.h1_weight)
                # Relu, [node_cnt, reg_hidden1]
                last_output = self.act(hidden)
                # [node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]
                # last_output_hidden = tf.matmul(last_output1, h2_weight)
                # last_output = tf.nn.relu(last_output_hidden)

            # [node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
            rep_aux = torch_sparse.spmm(rep_global[l]['index'], rep_global[l]['value'],\
                rep_global[l]['m'], rep_global[l]['n'], aux_input[:,l,:])
            for j in range(0, len(input_graph[l].edge_list), 2):
                edge_pair_1 = input_graph[l].edge_list[j][0]
                edge_pair_2 = input_graph[l].edge_list[j][1]
                req_re= torch.mul(rep_aux[edge_pair_1], rep_aux[edge_pair_2])
                req_result[l].append(req_re)
            req_result[l] = torch.stack(req_result[l])
            # rep_aux = torch.matmul(rep_global, aux_input)

            # if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
            # if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
            last_output = torch.concat([last_output, req_result[l]], 1)

            # if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
            # f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
            q_on_all = torch.matmul(last_output, self.last_w)
            q[l] += q_on_all
        return q
