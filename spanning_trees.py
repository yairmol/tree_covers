import math
import networkx as nx
from itertools import combinations

# from diamond_graphs import diamond_graph
from metric_spaces import MetricSpace, calculate_distortion, calculate_2tree_embedding_distortion


def all_different_spanning_trees(g: nx.Graph):
    num_edges_to_remove: int = len(g.edges) - (len(g.nodes) - 1)
    trees = list()
    e_choose_k = combinations(g.edges, num_edges_to_remove)
    for edges in e_choose_k:
        t = nx.Graph(g)
        for e in edges:
            t.remove_edge(e[0], e[1])
        if nx.is_tree(t):
            trees.append(t)
    return trees


def find_min_embedding(g: nx.Graph):
    trees = all_different_spanning_trees(g)
    tree_pairs = combinations(trees, 2)
    min_dist = 1000
    min_t1, min_t2 = None, None
    max_dist = 1
    for t1, t2 in tree_pairs:
        dist = calculate_2tree_embedding_distortion(g, t1, t2)
        if dist == 1.0:
            print("found isometric embedding")
            return t1, t2
        max_dist = max(max_dist, dist)
        if dist < min_dist:
            print("minimized dist", dist)
            min_t1 = t1
            min_t2 = t2
        min_dist = min(min_dist, dist)
    return min_t1, min_t2, min_dist


# d_2 = diamond_graph(2)
# print(d_2.nodes)
# tree1, tree2, d = find_min_embedding(d_2)
# nx.write_gexf(tree1, "tree1.gexf")
# nx.write_gexf(tree2, "tree2.gexf")
# print("minimum distortion", d)
tree1 = nx.read_gexf("graphs/tree1.gexf")
t_1 = nx.Graph()
t_1.add_edges_from([(eval(e[0]), eval(e[1])) for e in tree1.edges])
tree2 = nx.read_gexf("graphs/tree2.gexf")
t_2 = nx.Graph()
t_2.add_edges_from([(eval(e[0]), eval(e[1])) for e in tree2.edges])
print(t_1.nodes)
dist = calculate_2tree_embedding_distortion(diamond_graph(2), t_1, t_2)
print(dist)
