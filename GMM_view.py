from GMM import GMM
import numpy as np
import networkx as nx
from pymnet import *
import os
import matplotlib.pyplot as plt



def find_connected_components(graph):
    connected_components = []
    # print(graph.nodes)
    # print(type(graph.nodes))
    for node in graph.nodes:
        connected_component = nx.node_connected_component(graph, node)
        if connected_component not in connected_components:
            connected_components.append(connected_component)
    return connected_components


def m_draw(G1, G2,rand_g, node_num, type, prev_solutions, times):

    N = node_num
    net = MultilayerNetwork(aspects=1)
    M = 2

    # 设置层的背景颜色

    # 创建节点
    for j in range(0, N):
        net.add_node(j)
    # 创建层
    layer_name = ['layer1', 'layer2']
    for j in range(M):
        net.add_layer(layer_name[j])

    edges = list(G1.edges())
    for edge in edges:
        node1, node2 = edge
        if node1 not in prev_solutions and node2 not in prev_solutions:
            net[node1, node2, 'layer1', 'layer1'] = 1
    edges = list(G2.edges())
    for edge in edges:
        node1, node2 = edge
        if node1 not in prev_solutions and node2 not in prev_solutions:
            net[node1, node2, 'layer2', 'layer2'] = 1
    # 创建层间的边
    for j in range(0, N):
        net[j, j, 'layer1', 'layer2'] = 1
    colors = {}
    for node in net.iter_node_layers():
        if node[0] in prev_solutions:
            colors[node] = "red"
        else:
            colors[node] = "black"

    fig = draw(net, layout="circular", layershape="circle", layerPadding=0.2, azim=-60, elev=15,
               figsize=(10, 10),
               defaultEdgeWidth=0.3,layerColorDict={"layer1":"#d1e9ff","layer2":"#f5ffc7"},
               autoscale=True, nodeColorDict=colors, nodeSizeRule={"rule":"degree","propscale":0.02})
    #dir = f'results/{dataname}/{dataname}-{num0}-{num1}/{type}/'
    dir = f'random_graph/{node_num}/{type}/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(
        dir + f'/g为{rand_g}_{times}图.png')  # 保存图片
    plt.close(fig)

    adjacent_1 = nx.to_pandas_adjacency(G1)
    adjacent_1.to_csv(dir+f"adjacency_matrix_1_{times}.csv")
    adjacent_2 = nx.to_pandas_adjacency(G2)
    adjacent_2.to_csv(dir+f"adjacency_matrix_2_{times}.csv")


if __name__ == "__main__":


    for times in range(1):
        # max_n = 15
        # min_n = 8
        # cur_n = np.random.randint(max_n - min_n + 1) + min_n
        # rand_g = np.random.rand()

        first_graph = nx.Graph()
        second_graph = nx.Graph()

        link1, link2 = GMM(7, 0.5)
        for edge1,edge2 in link1:
            first_graph.add_edge(edge1, edge2)

        for edge1,edge2 in link2:
            second_graph.add_edge(edge1, edge2)

        m_draw(first_graph, second_graph,0.5, 7, '7nodes_with_0.5g', [], times)



