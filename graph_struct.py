# graph_struct.py

# LinkedTable类表示一个链表结构，可以动态地添加元素
class LinkedTable:
    def __init__(self):
        # 初始化节点数量为0
        self.n = 0
        # 初始化容量为0
        self.ncap = 0
        # 初始化链表头为空列表
        #self.head = []
        self.head = [[]]

        # add_entry方法用于向指定链表中添加内容

    def add_entry(self, head_id, content):
        """将内容添加到指定linked list中"""
        # 如果要添加的链表位置大于当前节点数量
        if head_id >= self.n:
            # 如果该位置加1大于当前容量
            if head_id + 1 > self.ncap:
                # 更新容量为原来的两倍或者达到要添加的位置，取二者中的较大值
                self.ncap = max(self.ncap * 2, head_id + 1)
                # 在当前容量和要添加的位置之间扩展空列表
                # self.head.extend([None] * (head_id + 1 - self.n))
                # self.head.extend([[]]*(head_id + 1 -self.n))
                self.head.extend([[] for _ in range(head_id + 1 - self.n)])
                # 更新节点数量为要添加的位置加1
            self.n = head_id + 1

            # 在指定链表位置添加内容
        self.head[head_id].append(content)
        # resize方法用于调整表格大小

    def resize(self, new_n):
        """调整表格大小"""
        # 如果新的容量大于当前容量
        if new_n > self.ncap:
            # 更新容量为原来的两倍或者达到新的容量，取二者中的较大值
            self.ncap = max(self.ncap * 2, new_n)
            # 在当前容量和新的容量之间扩展空列表
            #self.head.extend([None] * (new_n - len(self.head)))
            #self.head.extend([[]] * (new_n - len(self.head)))
            self.head.extend([[] for _ in range(new_n - len(self.head))])
            # 更新节点数量为新的容量
        self.n = new_n
        # 清空所有链表内容
        for entry in self.head:
            if entry is not None:
                entry.clear()

        # GraphStruct类表示图的结构，包含入边、出边、子图和边列表等信息


class GraphStruct:
    def __init__(self):
        # 初始化出边表、入边表、子图表和边列表等为空对象，并初始化节点数量、边数量和子图数量为0
        self.out_edges = LinkedTable()  # 出边表，表示每个节点的出边信息
        self.in_edges = LinkedTable()  # 入边表，表示每个节点的入边信息
        self.subgraph = LinkedTable()  # 子图表，表示每个子图的节点信息
        self.edge_list = []  # 边列表，表示所有边的信息
        self.num_nodes = 0  # 节点数量，表示图中节点的数量
        self.num_edges = 0  # 边数量，表示图中边的数量
        self.num_subgraph = 0  # 子图数量，表示图中子图的数量

    # add_edge方法用于向图中添加一条边，并更新相关的信息
    def add_edge(self, idx, x, y):
        """添加一条边"""
        # 将边的信息加入到出边表中，键是节点x，值是(idx, y)的元组对列表形式
        self.out_edges.add_entry(x, (idx, y))
        # 将边的信息加入到入边表中，键是节点y，值是(idx, x)的元组对列表形式（因为是无向图）
        self.in_edges.add_entry(y, (idx, x))
        # 更新边的数量加1
        self.num_edges += 1
        # 将边的信息加入到边列表中，形式为(x, y)的元组对列表形式
        self.edge_list.append((x, y))
        # 断言确保边的数量与边列表的长度相等（用于调试）
        assert self.num_edges == len(self.edge_list)
        # 断言确保边的索引值与当前边的数量减1相等（用于调试）
        assert self.num_edges - 1 == idx
        # add_node方法用于向图中添加一个节点，并更新相关的信息（这里是子图的节点

    def resize(self, num_subgraph, num_nodes=0):
        self.num_nodes = num_nodes
        self.num_edges = 0
        self.edge_list = []
        self.num_subgraph = num_subgraph
        self.in_edges.resize(self.num_nodes)
        self.out_edges.resize(self.num_nodes)
        self.subgraph.resize(self.num_subgraph)

    def add_node(self,subg_id,n_idx):
        self.subgraph.add_entry(subg_id,n_idx)