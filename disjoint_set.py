# 并查集类，用于处理集合的合并与查询操作
class DisjointSet:
    # 初始化方法，创建并查集，包含集合大小和初始的秩（rank）
    def __init__(self, graphSize):
        # union_set 用于存储每个节点的父节点，初始化时每个节点的父节点都是自己
        self.union_set = [0] * graphSize
        for i in range(graphSize):
            self.union_set[i] = i
        # rank_count 用于存储每个集合（以根节点为代表）的秩，初始秩都为1
        self.rank_count = [1] * graphSize
        # max_rank_count 记录当前并查集中秩最大的集合的秩
        self.max_rank_count = 1
        # ccd_score
        self.ccd_score = 0.0
        # 查找根节点的方法，同时进行路径压缩优化

    def find_root(self, node):
        # 如果当前节点不是根节点（即它的父节点不是自己）
        if node != self.union_set[node]:
            # 递归查找根节点，并将路径上的所有节点直接连接到根节点上
            rootNode = self.find_root(self.union_set[node])
            self.union_set[node] = rootNode  # 路径压缩
            return rootNode
        else:
            # 当前节点是根节点
            return node


    def merge(self, node1, node2):
        # 分别找到两个节点所在的集合的根节点
        node1_root = self.find_root(node1)
        node2_root = self.find_root(node2)
        maxn = 0
        # 如果两个节点不在同一个集合中
        if node1_root != node2_root:
            # 获取两个集合的秩
            node1_rank = self.rank_count[node1_root]
            node2_rank = self.rank_count[node2_root]
            temp1 = node1_rank * (node1_rank - 1) / 2.0 + node2_rank * (node2_rank - 1) / 2.0
            # 更新ccd_score
            self.ccd_score -= temp1
            temp2 = (node1_rank + node2_rank) * (node1_rank + node2_rank - 1) / 2.0
            self.ccd_score += temp2

            # 根据秩的大小来合并集合，秩小的集合合并到秩大的集合中
            if node2_rank > node1_rank:
                self.union_set[node1_root] = node2_root  # 将node1的根节点连接到node2的根节点上
                self.rank_count[node2_root] += self.rank_count[node1_root]  # 更新秩
                # 如果合并后的集合的秩超过了当前最大的秩，则更新最大秩
                if self.rank_count[node2_root] > self.max_rank_count:
                    self.max_rank_count = self.rank_count[node2_root]
            else:
                self.union_set[node2_root] = node1_root  # 将node2的根节点连接到node1的根节点上
                self.rank_count[node1_root] += self.rank_count[node2_root]  # 更新秩
                # 如果合并后的集合的秩超过了当前最大的秩，则更新最大秩
                if self.rank_count[node1_root] > self.max_rank_count:
                    self.max_rank_count = self.rank_count[node1_root]

    def get_biggest_component_current_ratio(self) -> float:
        # 返回最大秩的集合大小与整个并查集大小的比例
        print("len.union_set:{},graphsize:{}".format(len(self.union_set),len(self.rank_count)))
        return self.max_rank_count / len(self.union_set)

        # 获取指定根节点的秩

    def get_rank(self, rootNode) -> int:
        # 返回指定节点的秩
        return self.rank_count[rootNode]