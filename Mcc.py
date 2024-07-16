#coding=utf-8
import networkx as nx
def find_connected_components(graph):
    connected_components = []
    idx_map = [-1] * len(graph.nodes)
    for node in graph.nodes:
        if idx_map[node] >=0 :
            continue
        connected_component = nx.node_connected_component(graph, node) #获取和某个节点相连的节点
        for n in connected_component:
            idx_map[n] = 0
        if connected_component not in connected_components:
            connected_components.append(connected_component)
    return connected_components


def find_integer_in_sets(integer, set_list):
    for index, integer_set in enumerate(set_list):
        if integer in integer_set:
            return index  # 返回整数所在的集合索引

def deledge(graph, connected_components, remove_edge):
    for (u, v) in graph.edges:
        # print(find_integer_in_sets(u, connected_components1))
        #如果u,v不在同一个联通分量的话就删除u,v的连边
        if v not in connected_components[find_integer_in_sets(u, connected_components)]:
            graph.remove_edge(u, v)
            remove_edge.add((u,v))
            remove_edge.add((v,u))

def find_max_set_length(set_list):
    max_length = 0
    for integer_set in set_list:
        set_length = len(integer_set)
        if set_length > max_length:
            max_length = set_length
    return max_length
def find_max_edge_number(set_list):
    max_length = 0
    max_edge = []
    for integer_set in set_list:
        set_length = len(integer_set)
        if set_length > max_length:
            max_length = set_length
            max_edge = integer_set
    return max_edge

def find_set_length(set_list):
    set_lengths = []
    for integer_set in set_list:
        set_length = len(integer_set)
        set_lengths.append(set_length)
    return set_lengths

def find_isolated_node(set_list):
    isolated_nodes_indices = []
    for idx, component in enumerate(set_list):
        if len(component) == 1:
            isolated_nodes_indices.append(component)
    return isolated_nodes_indices

def MCC(G1, G2,remove_edge):
    #返回每个节点与其他节点联通的列表
    connected_components1 = find_connected_components(G1)
    connected_components2 = find_connected_components(G2)
    while connected_components1 != connected_components2:
        deledge(G2, connected_components1,remove_edge[1])
        connected_components2 = find_connected_components(G2)
        deledge(G1, connected_components2,remove_edge[0])
        connected_components1 = find_connected_components(G1)
    return connected_components1 , remove_edge



