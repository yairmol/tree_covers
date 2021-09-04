import os
from datetime import datetime
from typing import Tuple, Any, Callable, Set, Union, Dict

import networkx as nx

from metric_spaces import calculate_2tree_embedding_distortion, calc_2tree_embedding_distortion_with_known_distances, \
    _graph_distances, calc_2trees_embedding_distortion_low_memory, calc_2trees_embedding_distortion_medium_memory

Vertex = Any
Edge = Tuple[Vertex, Vertex]


class DiamondGraph:

    nodes_mapping: Dict[Any, int] = {1: 1, 2: 2, 3: 3, 4: 4}
    last_node = 4

    def __init__(self, k, graph_data=None):
        self.k = k
        if graph_data is not None:
            self.graph = graph_data['graph']
            self.one = graph_data['one']
            self.two = graph_data['two']
            self.three = graph_data['three']
            self.four = graph_data['four']
        else:
            self.one = 1
            self.two = 2
            self.three = 3
            self.four = 4
            self.graph = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])
            for i in range(k - 1):
                self.graph = self._create_next_diamond_graph(self.graph)

    @staticmethod
    def size_v_k(k):
        return 4 + ((2 * ((4 ** k) - 4)) / 3)

    @staticmethod
    def map_vertex(u, v, j, transform_one=False) -> int:
        size_V_i = 4
        four_to_power_i_minus_1 = 1
        u_power, v_power = 1, 1
        if v < u:
            u, v = v, u
        if transform_one:
            u = 5
        else:
            while u > size_V_i:
                u_power = size_V_i
                v_power = size_V_i
                four_to_power_i_minus_1 *= 4
                size_V_i = size_V_i + 2 * four_to_power_i_minus_1
        while v > size_V_i:
            v_power = size_V_i
            four_to_power_i_minus_1 *= 4
            size_V_i = size_V_i + 2 * four_to_power_i_minus_1
        print(u, u_power, v, v_power, size_V_i)
        return (u - u_power) + (v - v_power) + size_V_i + j

    @staticmethod
    def _create_next_diamond_graph(g_k: nx.Graph, to_enumerate=False) -> nx.Graph:
        g_k_plus_1 = nx.Graph()
        for node in g_k.nodes:
            g_k_plus_1.add_node(node)
        for edge in g_k.edges:
            v_e0, v_e1 = (edge, 0), (edge, 1)
            if to_enumerate:
                v_e0 = DiamondGraph.last_node + 1
                DiamondGraph.nodes_mapping[(frozenset(edge), 0)] = v_e0
                v_e1 = v_e0 + 1
                DiamondGraph.nodes_mapping[(frozenset(edge), 1)] = v_e1
                DiamondGraph.last_node = v_e1
            g_k_plus_1.add_node(v_e0)
            g_k_plus_1.add_node(v_e1)
            g_k_plus_1.add_edge(edge[0], v_e0)
            g_k_plus_1.add_edge(edge[0], v_e1)
            g_k_plus_1.add_edge(edge[1], v_e0)
            g_k_plus_1.add_edge(edge[1], v_e1)
        return g_k_plus_1

    def next_diamond_graph(self, to_enumerate=False):
        next_graph = self._create_next_diamond_graph(self.graph, to_enumerate)
        return DiamondGraph(self.k + 1, {'graph': next_graph, 'one': self.one, 'two': self.two,
                                         'three': self.three, 'four': self.four})

    def four_subgraphs(self):
        g = nx.Graph(self.graph)
        main_vertices = ['one', 'two', 'three', 'four']
        d_k_is = []
        for i in range(4):
            d_k_i = nx.Graph()
            path = nx.shortest_path(g, getattr(self, main_vertices[i]), getattr(self, main_vertices[(i + 1) % 4]))
            length = len(path)
            middle_nodes = set()
            while len(path) == length:
                middle_nodes.add(path[int((length - 1)/2)])
                edges = [(path[i], path[i+1]) for i in range(length - 1)]
                d_k_i.add_edges_from(edges)
                g.remove_edges_from(edges)
                if nx.has_path(g, getattr(self, main_vertices[i]), getattr(self, main_vertices[(i + 1) % 4])):
                    path = nx.shortest_path(g, getattr(self, main_vertices[i]), getattr(self, main_vertices[(i + 1) % 4]))
                else:
                    break
            middle_nodes = list(middle_nodes)
            one = getattr(self, main_vertices[i]) if i % 2 == 0 else getattr(self, main_vertices[(i + 1) % 4])
            three = getattr(self, main_vertices[(i + 1) % 4]) if i % 2 == 0 else getattr(self, main_vertices[i])
            d_k_i = DiamondGraph(self.k - 1,
                                 {
                                     "graph": d_k_i,
                                     "one": one,
                                     "two": middle_nodes[0],
                                     "three": three,
                                     "four": middle_nodes[1]
                                 })
            d_k_is.append(d_k_i)
        return d_k_is

    def spanning_tree(self):
        if self.k == 1:
            t1 = nx.Graph(self.graph)
            t1.remove_edge(self.one, self.two)
            return t1
        d_k_is = self.four_subgraphs()
        t_is = [d_k_i.spanning_tree() for d_k_i in d_k_is]
        t1 = nx.Graph()
        for t_i in t_is:
            t1.add_edges_from(t_i.edges)
        t1.remove_edge(self.one, list(t1[self.one].keys())[0])
        return t1

    def two_spanning_trees(self):
        if self.k == 1:
            t1 = nx.Graph(self.graph)
            t2 = nx.Graph(self.graph)
            t1.remove_edge(self.one, self.two)
            t2.remove_edge(self.two, self.three)
            return t1, t2
        d_k_is = self.four_subgraphs()
        t_is = [d_k_i.two_spanning_trees() for d_k_i in d_k_is]
        t1 = nx.Graph()
        t2 = nx.Graph()
        for t_1_i, t_2_i in t_is:
            t1.add_edges_from(t_1_i.edges)
            t2.add_edges_from(t_2_i.edges)
        if self.k != 3:
            p1 = nx.shortest_path(t1, self.one, self.two)
            t1.remove_edge(self.one, p1[1])
            p2 = nx.shortest_path(t2, self.two, self.three)
            t2.remove_edge(self.two, p2[1])
        return t1, t2


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


def create_trees_embedding(g_k: nx.Graph, tree1):
    tree2 = nx.Graph(g_k)
    missing_edges = set(g_k.edges).difference(set(tree1.edges))
    for edge in missing_edges:
        opposite_edge = (edge[0], (edge[1][0], 1 - edge[1][1]))
        tree2.remove_edge(opposite_edge[0], opposite_edge[1])
    print([e for e in tree1.edges])
    return tree1, tree2


def calculate_2tree_embedding_and_distortion(g_k):
    t1, t2 = create_trees_embedding(g_k)
    dist = calculate_2tree_embedding_distortion(g_k, t1, t2)
    print(dist)
    return t1, t2, dist


def d2_spanning_trees():
    d2 = DiamondGraph(2).graph
    t1 = nx.Graph(d2)
    t1.remove_edges_from([(1, ((1, 2), 0)), (2, ((2, 3), 1)), (4, ((3, 4), 1)), (1, ((1, 4), 0)), (1, ((1, 2), 1))])
    t2 = nx.Graph(d2)
    t2.remove_edges_from([(2, ((1, 2), 0)), (3, ((2, 3), 0)), (3, ((3, 4), 1)), (4, ((3, 4), 0)), (4, ((1, 4), 0))])
    return t1, t2


def find_best_embedding(k: int, threshold: float):
    g = DiamondGraph(k)
    c1, c2 = g.two_spanning_trees()
    p_g = dict(nx.all_pairs_shortest_path(g.graph))
    print(nx.is_tree(c1), nx.is_tree(c2))
    print("cycles dist:", calc_2tree_embedding_distortion_with_known_distances(p_g, c1, c2))
    cycle1 = []
    cycle2 = []
    for i in range(1, 5):
        if i != 1:
            cycle1.remove(cycle1[len(cycle1) - 1])
            cycle2.remove(cycle2[len(cycle2) - 1])
        cycle1.extend(nx.shortest_path(c1, i, i % 4 + 1))
        cycle2.extend(nx.shortest_path(c2, i, i % 4 + 1))
    edges1 = [(cycle1[i], cycle1[i+1]) for i in range(len(cycle1) - 1)]
    edges2 = [(cycle2[i], cycle2[i+1]) for i in range(len(cycle2) - 1)]
    print(len(edges1), len(edges2))
    min_distortion = 100
    min_t1 = None
    min_t2 = None
    for e1 in edges1:
        for e2 in edges2:
            c1.remove_edge(e1[0], e1[1])
            c2.remove_edge(e2[0], e2[1])
            dist = calc_2tree_embedding_distortion_with_known_distances(p_g, c1, c2)
            if dist < min_distortion and dist < threshold:
                min_t1 = nx.Graph(c1)
                min_t2 = nx.Graph(c2)
            min_distortion = min(dist, min_distortion)
            print(dist)
            c1.add_edge(e1[0], e1[1])
            c2.add_edge(e2[0], e2[1])
    return min_distortion, min_t1, min_t2


def dg_two_trees_cover(k: int, ext_existing_edge: Callable[[Edge, int], Set[Edge]],
                       ext_ne_edge: Callable[[Edge, int], Set[Edge]]) -> (DiamondGraph, nx.Graph, nx.Graph):
    def extend_tree(t: nx.Graph, d: DiamondGraph, t_num: int):
        edges = set(t.edges)
        for edge in edges:
            t.remove_edge(edge[0], edge[1])
            t.add_edges_from(ext_existing_edge(edge, t_num))
        for edge in set(d.graph.edges).difference(set(t.edges)):
            t.add_edges_from(ext_ne_edge(edge, t_num))
    d_k: DiamondGraph = DiamondGraph(1)
    t1, t2 = nx.Graph(d_k.graph), nx.Graph(d_k.graph)
    t1.remove_edge(1, 2)
    t2.remove_edge(2, 3)
    for i in range(1, k):
        extend_tree(t1, d_k, 1)
        extend_tree(t2, d_k, 2)
        d_k = d_k.next_diamond_graph()
        t1.add_nodes_from(d_k.graph.nodes)
        t2.add_nodes_from(d_k.graph.nodes)
    return d_k, t1, t2


def dg2_best_spanning_trees():
    d2 = DiamondGraph(2).graph
    t1 = nx.Graph(d2)
    t2 = nx.Graph(d2)
    t1.remove_edges_from([(2, ((1, 2), 0)), (2, ((1, 2), 1)), (2, ((2, 3), 0)), (3, ((3, 4), 0)), (4, ((1, 4), 0))])
    t2.remove_edges_from([(1, ((1, 2), 0)), (3, ((2, 3), 0)), (3, ((2, 3), 1)), (4, ((3, 4), 0)), (1, ((1, 4), 0))])
    return d2, t1, t2


def node_depth(node: Union[tuple, int]):
    return 0 if not isinstance(node, tuple) else 1 + max(node_depth(node[0][0]), node_depth(node[0][1]))


class Node:
    def __init__(self, data):
        self.data = data

    def __lt__(self, other):
        self_nd = node_depth(self.data)
        other_nd = node_depth(other.data)
        if self_nd < other_nd:
            return True
        if self_nd == other_nd:
            return self.data < other.data
        return False


def two_spanning_trees_by_construction(k: int, to_enumerate=True):
    """create two spanning trees of the diamond graph D_k.
    at each iteration choose which edges to add to each spanning tree,
    according to the edges that are in the previous spanning tree.
    assumption: for every edge e=(u,v) we assume u < v where < means
    that u was generated before v (or that u and v are both in V1 and then they are both integers)"""
    start, d1 = 2, DiamondGraph(1)
    t1, t2 = nx.Graph(d1.graph), nx.Graph(d1.graph)
    t1.remove_edge(1, 2)
    t2.remove_edge(3, 4)
    d_i_minus_1, d_i = d1, d1
    # if k > 2:
    #     d_i_minus_1, d_i = d1, d1.next_diamond_graph(to_enumerate=to_enumerate)
    #     # TODO: start from a better trees pair3
    #     t1, t2, start = get_initial_trees(), start + 1
    for i in range(2, k + 1):
        d_i = d_i_minus_1.next_diamond_graph(to_enumerate=to_enumerate)
        t1.add_nodes_from(d_i.graph.nodes)
        t2.add_nodes_from(d_i.graph.nodes)
        for (u, v) in d_i_minus_1.graph.edges:
            if Node(v) < Node(u):
                u, v = v, u
            e = (u, v)
            v_e0, v_e1 = (e, 0), (e, 1)
            if to_enumerate:
                v_e0 = DiamondGraph.nodes_mapping[(frozenset(e), 0)]
                v_e1 = DiamondGraph.nodes_mapping[(frozenset(e), 1)]
            if (u, v) in t1.edges and (u, v) in t2.edges:
                t1.add_edges_from([(u, v_e1), (v, v_e0), (v, v_e1)])
                t2.add_edges_from([(u, v_e0), (u, v_e1), (v, v_e0)])
                t1.remove_edge(u, v)
                t2. remove_edge(u, v)
            elif (u, v) in t1.edges:
                t1.add_edges_from([(u, v_e1), (v, v_e0), (v, v_e1)])
                t2.add_edges_from([(u, v_e0), (u, v_e1)])
                t1.remove_edge(u, v)
            elif (u, v) in t2.edges:
                t1.add_edges_from([(u, v_e0), (u, v_e1)])
                t2.add_edges_from([(u, v_e0), (v, v_e0), (v, v_e1)])
                t2.remove_edge(u, v)
        d_i_minus_1 = d_i
    DiamondGraph.nodes_mapping.clear()
    return d_i.graph, t1, t2


def reverse_mapping(d: dict):
    return {v: k for k, v in d.items()}


def main(size, write_graphs=False):
    print(datetime.now(), "finding spanning trees")
    d, t1, t2 = two_spanning_trees_by_construction(size)
    print(datetime.now(), "found")
    if write_graphs:
        path = f"./graph_files"
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        nx.write_gexf(d, f"{path}/dg_{size}.gexf")
        nx.write_gexf(t1, f"{path}/t1_{size}.gexf")
        nx.write_gexf(t2, f"{path}/t2_{size}.gexf")
    # print("missing edges in t1:", set(d.edges).difference(t1.edges))
    # print("missing edges in t2:", set(d.edges).difference(t2.edges))
    print("checking correctness:", nx.is_tree(t1), nx.is_tree(t2))
    print(datetime.now(), "calculating distortion")
    print(calc_2trees_embedding_distortion_medium_memory(d, t1, t2, 1))
    print(datetime.now(), "finished")


if __name__ == "__main__":
    main(2, True)
