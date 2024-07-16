import  random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def add_edges_from_file(graph, file):
    for line in file:
        edge = line.strip().split(' '.encode())
        u, v = int(edge[0]), int(edge[1])
        graph.add_edge(u, v)
    return graph

def find_connected_components(graph):
    connected_components = []
    # print(graph.nodes)
    # print(type(graph.nodes))
    for node in graph.nodes:
        connected_component = nx.node_connected_component(graph, node)
        if connected_component not in connected_components:
            connected_components.append(connected_component)
    return connected_components

def find_integer_in_sets(integer, set_list):
    for index, integer_set in enumerate(set_list):
        if integer in integer_set:
            return index  # 返回整数所在的集合索引

def deledge(graph, connected_components):
    for (u, v) in graph.edges:
        # print(find_integer_in_sets(u, connected_components1))
        if v not in connected_components[find_integer_in_sets(u, connected_components)]:
            graph.remove_edge(u, v)

def find_max_set_length(set_list):
    max_length = 0
    for integer_set in set_list:
        set_length = len(integer_set)
        if set_length > max_length:
            max_length = set_length

    return max_length

def MCC(G1, G2):
    connected_components1 = find_connected_components(G1)
    connected_components2 = find_connected_components(G2)
    while connected_components1 != connected_components2:
        deledge(G2, connected_components1)
        connected_components2 = find_connected_components(G2)
        deledge(G1, connected_components2)
        connected_components1 = find_connected_components(G1)
    M = find_max_set_length(connected_components1)
    M2 = find_max_set_length(connected_components2)
    return G1,G2, M

def draw_anc(G1, G2, MCC1, MCC2, num):
    MCC = G1.number_of_edges() + G2.number_of_edges()
    print("edges", MCC)
    # 计算填充数量
    fill_count_1 = MCC - len(MCC1)
    # 扩展MCC1的长度并填充
    if fill_count_1 > 0:
        fill_value = MCC1[-1]  # 使用MCC1的最后一个值填充
        MCC1 = np.append(MCC1, np.full(fill_count_1, fill_value))
    random_score = [sum(i / (len(G1.edges()) + len(G2.edges())) for i in MCC1)]
    x1 = np.ones(len(MCC1))
    for i in range(0, len(MCC1)):
        x1[i] = i / len(MCC1)

    fill_count_2 = MCC - len(MCC2)
    # 扩展MCC1的长度并填充
    if fill_count_2 > 0:
        fill_value = MCC2[-1]  # 使用MCC1的最后一个值填充
        MCC2 = np.append(MCC2, np.full(fill_count_2, fill_value))
    finder_score = [sum(i / (len(G1.edges()) + len(G2.edges())) for i in MCC2)]

    x2 = np.ones(len(MCC2))
    for i in range(0, len(MCC2)):
        x2[i] = i / len(MCC2)

    plt.figure(figsize=(10,5))
    plt.plot(x1, MCC1, label=f'random_anc:{random_score}')
    plt.plot(x2, MCC2, label=f'finder_anc:{finder_score}')
    plt.ylabel('Value')
    plt.title(f'{num}_nodes')
    plt.legend()
    # 设置 x 轴范围为 0 到 0.2
    plt.xlim(0, 1)
    plt.savefig(f'./compare_random/{num}_img_best.png')


if __name__ == "__main__" :
    # plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    # plt.rcParams["axes.unicode_minus"] = False
    node_num = 10
    adj1 = np.load(f"./data/syn_30/{node_num}/adj1_%s.npy" % 101)
    adj2 = np.load(f"./data/syn_30/{node_num}/adj2_%s.npy" % 101)
    G1 = nx.from_numpy_array(adj1)
    original_edge = len(G1.edges())
    G2 = nx.from_numpy_array(adj2)

    edge_sequence = []
    with open(f'./data/syn_30/{node_num}/edge_sequence', 'r') as file:
        for j in file:
            j = eval(j)
            edge_sequence.append(j)
    random_del_list = []
    edge_sequence_dic = {}
    for j in edge_sequence[0]:
        edge_sequence_dic.setdefault(j, edge_sequence[1][j])

    G1, G2, M_original_max= MCC(G1, G2)
    M_ori = M_original_max
    sequence = []
    for edge_1, edge_2 in G1.edges:
        if (edge_1, edge_2) in edge_sequence[1]:
            keys = list(edge_sequence_dic.keys())[list(edge_sequence_dic.values()).index((edge_1, edge_2))]
            if keys < original_edge:
                sequence.append(keys)
    for edge_1, edge_2 in G2.edges:
        if (edge_1, edge_2) in edge_sequence[1]:
            keys = list(edge_sequence_dic.keys())[list(edge_sequence_dic.values()).index((edge_1, edge_2))]
            if keys >= original_edge:
                sequence.append(keys)
    edge_sequence[0] = sequence
    random_MaxCClist = []

    while M_ori > 1:
        index_1 = random.choice(edge_sequence[0])

        if index_1 < original_edge:
            G1.remove_edge(edge_sequence[1][index_1][0], edge_sequence[1][index_1][1])
        else:
            G2.remove_edge(edge_sequence[1][index_1][0], edge_sequence[1][index_1][1])
        G1, G2, M_ori = MCC(G1, G2)
        sequence = []
        for edge_1, edge_2 in G1.edges:
            if (edge_1, edge_2) in edge_sequence[1]:
                keys = list(edge_sequence_dic.keys())[list(edge_sequence_dic.values()).index((edge_1, edge_2))]
                if keys < original_edge:
                    sequence.append(keys)
        for edge_1, edge_2 in G2.edges:
            if (edge_1, edge_2) in edge_sequence[1]:
                keys = list(edge_sequence_dic.keys())[list(edge_sequence_dic.values()).index((edge_1, edge_2))]
                if keys >= original_edge:
                    sequence.append(keys)
        edge_sequence[0] = sequence
        random_del_list.append(index_1)
        random_MaxCClist.append(M_ori/M_original_max)

    # print(2)
    # print("random", len(random_del_list))
    # print("random", random_MaxCClist)
    G1 = nx.from_numpy_array(adj1)
    G2 = nx.from_numpy_array(adj2)

    finder_MaxCClist = np.loadtxt(f'./data/syn_30/{node_num}/score_0', usecols=(0,))
    finder_list = np.loadtxt(f'./data/syn_30/{node_num}/soulation_0', usecols=(0,))
    finder_MaxCClist = finder_MaxCClist[1:]
    # print("finder",len(finder_list))
    # print("finder", MCC3)
    draw_anc(G1, G2, random_MaxCClist, finder_MaxCClist, node_num)


