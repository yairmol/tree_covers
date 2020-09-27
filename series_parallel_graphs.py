from typing import List, Optional, Union

import networkx as nx


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

    def __init__(self, s=None, t=None, **attr):
        super().__init__(**attr)
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


class DiamondGraph:
    def __init__(self, k):
        self.graph = SPGraph.K_2()
        for i in range(k):
            self.graph = self.next_diamond_graph()

    def next_diamond_graph(self):
        return (self.graph * self.graph) + (self.graph * self.graph)




d3 = DiamondGraph(3)
nx.write_gexf(d3.graph, "D3.gexf")

