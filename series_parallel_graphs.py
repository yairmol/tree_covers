from typing import List, Optional, Union, Any

import networkx as nx
# from diamond_graphs import diamond_graph
from metric_spaces import calculate_2tree_embedding_distortion


def un(node: Union[tuple, int], i: int):
    if isinstance(node, tuple):
        return tuple(list(node) + [i])
    return tuple([node, i])


class SPGraph(nx.Graph):
    @staticmethod
    def K_2():
        k2 = SPGraph(s=0, t=1)
        k2.add_edge(0, 1)
        return k2

    def __init__(self, s, t, incoming_graph_data: Any = None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self.s = s if s is not None else 0
        self.t = t if t is not None else 1
        self.add_nodes_from([self.s, self.t])

    # series composition
    def __mul__(self, other):
        nodes = set([un(n, 0) for n in self.nodes]).union(
            set([un(n, 1) for n in other.nodes if n != other.s]))
        sc = SPGraph(s=un(self.s, 0), t=un(other.t, 1))
        sc.add_nodes_from(nodes)
        # add edges of the first graph
        sc.add_edges_from([(un(u, 0), un(v, 0)) for u, v in self.edges])
        # add edges of second graph
        sc.add_edges_from([(un(u, 1), un(v, 1)) for u, v in other.edges if other.s not in [u, v]])
        # add edges the connect between the graphs
        sc.add_edges_from([(un(self.t, 0), un(v, 1)) for v in other[other.s].keys()])
        return sc

    # parallel composition
    def __add__(self, other):
        nodes = set([un(n, 0) for n in self.nodes]).union(
            set([un(n, 1) for n in other.nodes if n not in [other.s, other.t]]))
        pc = SPGraph(un(self.s, 0), un(self.t, 0))
        pc.add_nodes_from(nodes)
        pc.add_edges_from([(un(u, 0), un(v, 0)) for u, v in self.edges])
        pc.add_edges_from([(un(u, 1), un(v, 1)) for u, v in other.edges if len({u, v}.intersection({other.s, other.t})) == 0])
        pc.add_edges_from([(un(self.s, 0), un(v, 1)) for v in other[other.s].keys()])
        pc.add_edges_from([(un(u, 1), un(self.t, 0)) for u in other[other.t].keys()])
        return pc


class DiamondGraph(SPGraph):
    def __init__(self, k=None, sp_graph: SPGraph = None, **attr):
        if sp_graph is None:
            sp_graph = SPGraph(0, 2, nx.cycle_graph(4))
            for i in range(1, k):
                sp_graph = (sp_graph * sp_graph) + (sp_graph * sp_graph)
        super().__init__(s=sp_graph.s, t=sp_graph.t, incoming_graph_data=sp_graph, **attr)

    def next_diamond_graph(self):
        return DiamondGraph(sp_graph=(self * self) + (self * self))

    @staticmethod
    def two_trees(k: int):
        g = DiamondGraph(1)
        t1 = DiamondGraph(1)
        t1.remove_edge(0, 1)
        t2 = DiamondGraph(1)
        t2.remove_edge(1, 2)
        for i in range(1, k):
            t1 = (t1 * t1) + (t1 * t1)
            t1.remove_edge(t1.s, list(t1[t1.s].keys())[0])
            t2 = (t2 * t2) + (t2 * t2)
            t2.remove_edge(t2.t, list(t2[t2.t].keys())[0])
            g = g.next_diamond_graph()
        return t1, t2, g
