from typing import List
#no problem

# 定义一个名为GraphUtil的类
class GraphUtil:
    def __init__(self):
        pass

    # 删除指定节点的方法
    # def delete_node(self, adj_list_graph: List[List[int]], node: int):
    def delete_node(self, adj_list_graph, node):
        # 遍历指定节点相邻的所有节点
        for j in range(2):
            for i in range(len(adj_list_graph[j][node])):
                # 获取当前相邻的节点
                neighbour = adj_list_graph[j][node][i]
                # 从相邻节点的邻接列表中删除当前节点
                adj_list_graph[j][neighbour].remove(node)
            # 清空当前节点的邻接列表
            adj_list_graph[j][node].clear()


    # 恢复并添加节点的方法
    #def recover_add_node(self, backup_completed_adj_list_graph: List[List[int]], backup_all_vex: List[bool],adj_list_graph: List[List[int]], node: int, union_set: DisjointSet):
    def recover_add_node(self, backup_completed_adj_list_graph, backup_all_vex,
                            adj_list_graph, node, union_set):
        for i in range(2):
            # 遍历当前节点在备份邻接列表中的所有相邻节点
            for neighbor_node in backup_completed_adj_list_graph[i][node]:
                # 如果相邻节点尚未处理（即backupAllVex为True）
                if backup_all_vex[neighbor_node] :
                    # 调用addEdge方法添加边，并合并集合
                    #print(node,neighbor_node)
                    self.add_edge(adj_list_graph[i], node, neighbor_node)
                    union_set[i].merge(node, neighbor_node)
        backup_all_vex[node] = True


    # 添加边的方法
    def add_edge(self, adj_list_graph: List[List[int]], node0: int, node1: int):
        # 获取两个节点中的较大值作为最大节点编号
        max_node = max(node0, node1)
        # 如果邻接列表的长度小于最大节点编号+1，则扩展列表以容纳新的节点
        if len(adj_list_graph) - 1 < max_node:
            adj_list_graph.extend([] for _ in range(max_node - len(adj_list_graph) + 1))
            # 将两个节点添加到彼此的邻接列表中，表示它们之间存在一条边
        adj_list_graph[node0].append(node1)
        adj_list_graph[node1].append(node0)

