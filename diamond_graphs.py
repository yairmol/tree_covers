from typing import List, Tuple, Set, TypeVar, Callable, Type, Generic
from itertools import combinations
import networkx as nx
from series_parallel_graphs import SPGraph

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
Embedding = Callable[[T1], T2]


class MetricSpace(Generic[T]):
    def __init__(self, points: Set[T], d: Callable[[T, T], float]):
        self.points = points
        self.d = d


def create_next_diamond_graph(g_k):
    g_k_plus_1 = nx.Graph()
    new_nodes = set()
    for node in g_k.nodes:
        g_k_plus_1.add_node(node)
    for edge in g_k.edges:
        v_e0, v_e1 = (edge, 0), (edge, 1)
        new_nodes.add(v_e0)
        new_nodes.add(v_e1)
        g_k_plus_1.add_node(v_e0)
        g_k_plus_1.add_node(v_e1)
        g_k_plus_1.add_edge(edge[0], v_e0)
        g_k_plus_1.add_edge(edge[0], v_e1)
        g_k_plus_1.add_edge(edge[1], v_e0)
        g_k_plus_1.add_edge(edge[1], v_e1)
    return g_k_plus_1


def diamond_graph(k: int):
    g_i = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])
    for i in range(1, k):
        g_i = create_next_diamond_graph(g_i)
    return g_i


def get_source_edge(node: tuple):
    while isinstance(node, tuple):
        if isinstance(node[0], tuple):
            node = node[0]
        elif isinstance(node[1], tuple):
            node = node[1]
        else:
            return node


def diamond_graph_subgraph(g_k: nx.Graph, u):
    v = u % 4 + 1
    g_k_minus_1 = nx.Graph()
    g_k_minus_1.add_nodes_from([n for n in g_k.nodes if n in {u, v} or (isinstance(n, tuple) and get_source_edge(n) == (u, v))])
    g_k_minus_1.add_edges_from([(u, v) for u, v in g_k.edges if u in g_k_minus_1.nodes and v in g_k_minus_1.nodes])
    return g_k_minus_1


def depth(node):
    if not isinstance(node, tuple):
        return 0
    return max(depth(node[0]), depth(node[1])) + 1


def dg_spanning_tree(g_k: nx.Graph):
    def dg_spanning_tree_rec(g_k: nx.Graph) -> set:
        if len(g_k.nodes) == 4:
            ret = set(g_k.edges)
            ret.remove(list(g_k.edges)[0])
            return ret
        edges = set()
        for i in range(4):
            g_k_i = diamond_graph_subgraph(g_k, i + 1)
            print(len(g_k_i.nodes))
            t_i_edges = dg_spanning_tree_rec(g_k_i)
            if i == 0:
                n = list(g_k_i.nodes)[0]
                for node in g_k_i.nodes:
                    if depth(node) < depth(n):
                        n = node
                t_i_edges.remove((n, list(g_k_i[n].keys())[0]))
            edges.update(t_i_edges)
        return edges
    tree = nx.Graph()
    tree.add_edges_from(dg_spanning_tree_rec(g_k))
    return tree


def create_trees_embedding(g_k: nx.Graph):
    tree1 = dg_spanning_tree(g_k)
    tree2 = nx.Graph(g_k)
    missing_edges = set(g_k.edges).difference(set(tree1.edges))
    for edge in missing_edges:
        opposite_edge = (edge[0], (edge[1][0], 1 - edge[1][1]))
        tree2.remove_edge(opposite_edge[0], opposite_edge[1])
    print([e for e in tree1.edges])
    return tree1, tree2


def calculate_distortion(x_metric_space: MetricSpace[T1], y_metric_space: MetricSpace[T2], f: Embedding):
    distortion = 1.0
    for p1, p2 in combinations(x_metric_space.points, 2):
        # print(p1, p2)
        # print("X:", x_metric_space.d(p1, p2), "Y:", y_metric_space.d(f(p1), f(p2)))
        if y_metric_space.d(f(p1), f(p2)) / x_metric_space.d(p1, p2) > 1.0:
            print(p1, ",", p2, ",", f(p1), ",", f(p2), ",", x_metric_space.d(p1, p2), y_metric_space.d(f(p1), f(p2)))
        distortion = max(distortion, y_metric_space.d(f(p1), f(p2)) / x_metric_space.d(p1, p2))
    return distortion


def calculate_2tree_embedding_distortion(g, t1, t2):
    dist = calculate_distortion(MetricSpace(set(g.nodes), lambda u, v: len(nx.shortest_path(g, u, v)) - 1),
                                MetricSpace(set(t1.nodes), lambda u, v: min(len(nx.shortest_path(t1, u, v)),
                                                                            len(nx.shortest_path(t2, u, v))) - 1),
                                lambda x: x)
    return dist


def calculate_2tree_embedding_and_distortion(g_k):
    t1, t2 = create_trees_embedding(g_k)
    dist = calculate_2tree_embedding_distortion(g_k, t1, t2)
    print(dist)
    return t1, t2, dist


g3 = diamond_graph(3)
g2 = diamond_graph_subgraph(g3, 1)
print(g2.nodes)
print()
# t = dg_spanning_tree(g3)
# nx.write_gexf(g3, "t.gexf")
# g2 = diamond_graph_subgraph(g3, 1)
# print(g2.nodes)
# print(len(g2.nodes))
# nx.write_gexf(g2, "D2_subgraph.gexf")
