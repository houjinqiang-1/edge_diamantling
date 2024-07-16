# import torch
# a = torch.Tensor([[1, 2], [3, 4]])
# b = torch.Tensor([[5, 6], [7, 8]])
# hadamard_product = a * b
# print('hadamard_product:', hadamard_product)
#
# import torch
# # 检查是否有可用的 GPU，如果没有则使用 CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 定义 Y 节点的数量
# y_nodes_size = 3
# # 创建大小为 (y_nodes_size, 2) 的张量，并初始化为全1
# y_node_input = torch.ones((y_nodes_size, 2)).to(device)
# # 打印结果
# print("Y 节点输入张量：")
# print(y_node_input)

# import torch.nn.functional as F
# import torch
# # 假设 input_message 是一个大小为 (3, 4) 的示例输入矩阵
# input_message = torch.tensor([[0.1, 0.2, 0.3, 0.4],
#                                [0.5, 0.6, 0.7, 0.8],
#                                [0.9, 1.0, 1.1, 1.2]])
#
# # 假设我们的激活函数是 ReLU
# # 对 input_message 应用激活函数
# input_potential_layer = F.relu(input_message)
#
# # 打印结果
# print("激活后的输入矩阵：")
# print(input_potential_layer)

# import torch
# from torch_sparse import spmm
#
# # 假设输入的稀疏矩阵的参数
# n2nsum_param = {
#     0: {'index': torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),  # 稀疏矩阵的索引
#         'value': torch.tensor([0.5, 0.7, 0.3, 0.4]),          # 稀疏矩阵的权重值
#         'm': 3, 'n': 3}                                       # 稀疏矩阵的大小
# }
# # 假设输入的当前消息层
# cur_message_layer = torch.tensor([[1.0, 2.0],
#                                    [3.0, 4.0],
#                                    [5.0, 6.0]])
# # 打印当前消息层
# print("当前消息层：")
# print(cur_message_layer)
# # 进行稀疏矩阵乘法
# n2npool0 = spmm(n2nsum_param[0]['index'],  # 稀疏矩阵的索引
#                 n2nsum_param[0]['value'],  # 稀疏矩阵的权重值
#                 n2nsum_param[0]['m'],      # 稀疏矩阵的行数
#                 n2nsum_param[0]['n'],      # 稀疏矩阵的列数
#                 cur_message_layer)         # 当前消息层
#
# # 打印结果
# print("稀疏矩阵乘法的结果：")
# print(n2npool0)

# import torch
# # x1
# x1 = torch.tensor([[11,21,31],[21,31,41]],dtype=torch.int)
# x1.shape # torch.Size([2, 3])
# # x2
# x2 = torch.tensor([[12,22,32],[22,32,42]],dtype=torch.int)
# x2.shape  # torch.Size([2, 3])
# inputs = [x1, x2]
# aa = torch.cat(inputs, dim=0).shape
# print(aa)


# import torch
# # 假设 message_layer 是一个大小为 (2, 5, 3) 的三维张量
# message_layer = torch.randn(2, 5, 3)
# # 选择前3个节点的消息
# cur_message_layer = message_layer[:, 2:, :]
# # 选择从第3个节点到最后一个节点的消息
# y_cur_message_layer = message_layer[:, :2, :]
# # 对 cur_message_layer 和 y_cur_message_layer 进行 L2 归一化
# cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
# y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)
# # 打印结果
# print("cur_message_layer:\n", cur_message_layer)
# print("y_cur_message_layer:\n", y_cur_message_layer)

# import torch
# # 假设我们有两个张量
# cur_message_layer = torch.randn(2, 3, 4)  # 形状为 (batch_size=2, m=3, k=4)
# rep_y = torch.randn(2, 4, 5)              # 形状为 (batch_size=2, k=4, n=5)
#
# # 对 cur_message_layer 进行维度扩展
# cur_message_layer_expanded = torch.unsqueeze(cur_message_layer, dim=2)  # 形状为 (2, 3, 1, 4)
#
# # 对 rep_y 进行维度扩展
# rep_y_expanded = torch.unsqueeze(rep_y, dim=1)                           # 形状为 (2, 1, 4, 5)
#
# cur_message_layer_expanded_0 = torch.unsqueeze(cur_message_layer,dim=0)
# print("cur_message_layer_expanded的形状", cur_message_layer_expanded_0.shape)
# # 进行矩阵乘法
# result = torch.matmul(cur_message_layer_expanded.cuda(), rep_y_expanded.cuda())
#
# # 打印结果的形状
# print("结果的形状：", result.shape)


# import torch
#
# # 定义 temp1 和 cross_product
# temp1 = torch.rand(36, 64, 64)
# cross_product = torch.rand(64, 1)  # 形状为 (2, 3)
#
# # 定义 Shape1
# Shape1 = [36, 64]
#
# tiled_cross_product = torch.tile(cross_product, [Shape1[0], 1])
# print(tiled_cross_product)
#
# aa = torch.reshape(tiled_cross_product, [36, 64, 1])
#
# bb = torch.matmul(temp1, aa)
#
# cc = torch.reshape(bb, [36, 64])
#
#
#
#
# a = torch.rand(36, 1)
# b = torch.rand(36, 1)
# c = a + b


class Graph:
    def __init__(self):
        self.adj_list = [[1, 2], [0, 2], [0, 1]]
# 创建图实例
graph = Graph()
# 定义排除列表
not_chose_nodes = [0, 1, 2]
# 使用列表推导式和 any() 函数检查节点 0 的邻居列表中是否有任何邻居不在排除列表中
useful1 = any(neigh not in not_chose_nodes for neigh in graph.adj_list[0])
for neigh in graph.adj_list[0]:
    print(neigh)
# 打印结果
print(useful1)  # 输出 True，因为节点 0 的邻居 1 不在排除列表中



