import networkx as nx

from metric_spaces import calculate_2tree_embedding_distortion


class DiamondGraph:
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
                self.graph = self.create_next_diamond_graph(self.graph)

    @staticmethod
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
            t2.remove_edge(self.three, self.four)
            return t1, t2
        d_k_is = self.four_subgraphs()
        t_is = [d_k_i.two_spanning_trees() for d_k_i in d_k_is]
        t1 = nx.Graph()
        t2 = nx.Graph()
        for t_1_i, t_2_i in t_is:
            t1.add_edges_from(t_1_i.edges)
            t2.add_edges_from(t_2_i.edges)
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
    t1.remove_edges_from([(1, ((1, 2), 0)), (2, ((2, 3), 0)), (3, ((3, 4), 0)), (4, ((1, 4), 0)), (1, ((1, 2), 1))])
    t2 = nx.Graph(d2)
    t2.remove_edges_from([(2, ((1, 2), 0)), (3, ((2, 3), 0)), (3, ((3, 4), 0)), (4, ((1, 4), 1)), (3, ((3, 4), 1))])
    return t1, t2


def find_best_embedding(k: int):
    g = DiamondGraph(k)
    d_3_tree1, d_3_tree2 = g.two_spanning_trees()
    print(nx.is_tree(d_3_tree1), nx.is_tree(d_3_tree2))
    print(calculate_2tree_embedding_distortion(g.graph, d_3_tree1, d_3_tree2))
    cycle1 = []
    cycle2 = []
    for i in range(1, 5):
        if i != 1:
            cycle1.remove(cycle1[len(cycle1) - 1])
            cycle2.remove(cycle2[len(cycle2) - 1])
        cycle1.extend(nx.shortest_path(d_3_tree1, i, i % 4 + 1))
        cycle2.extend(nx.shortest_path(d_3_tree2, i, i % 4 + 1))
    edges1 = [(cycle1[i], cycle1[i+1]) for i in range(len(cycle1) - 1)]
    edges2 = [(cycle2[i], cycle2[i+1]) for i in range(len(cycle2) - 1)]
    print(len(edges1), len(edges2))
    for e1 in edges1:
        for e2 in edges2:
            d_3_tree1.remove_edge(e1[0], e1[1])
            d_3_tree2.remove_edge(e2[0], e2[1])
            print(calculate_2tree_embedding_distortion(g.graph, d_3_tree1, d_3_tree2))
            d_3_tree1.add_edge(e1[0], e1[1])
            d_3_tree2.add_edge(e2[0], e2[1])
    nx.write_gexf(d_3_tree1, "graphs/D3_tree1.gexf")
    nx.write_gexf(d_3_tree2, "graphs/D3_tree2.gexf")
    print(nx.is_tree(d_3_tree1) and nx.is_tree(d_3_tree2))
    print(calculate_2tree_embedding_distortion(g.graph, d_3_tree1, d_3_tree2))
