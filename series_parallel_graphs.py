from typing import List, Optional, Union, Any, Tuple, Callable
import networkx as nx
import numpy as np
Vertex = int
Edge = Tuple[Vertex, Vertex]


class SPGraph(nx.Graph):
    @staticmethod
    def k2():
        k2 = SPGraph(s=0, t=1)
        k2.add_edge(0, 1)
        return k2

    @staticmethod
    def d1():
        return SPGraph(s=1, t=3, incoming_graph_data={(1, 2), (2, 3), (3, 4), (4, 1)})

    def __init__(self, s, t, incoming_graph_data: Any = None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self.s = s if s is not None else 0
        self.t = t if t is not None else 1
        self.add_nodes_from([self.s, self.t])

    def __repr__(self):
        return f"s: {self.s}, t: {self.t} nodes: {self.nodes}, edges: {self.edges}"

    def __str__(self):
        return f"s: {self.s}, t: {self.t} nodes: {self.nodes}, edges: {self.edges}"

    def map_graph(self, f: Callable[[Vertex], Vertex]):
        return SPGraph(f(self.s), f(self.t), {(f(u), f(v)) for (u, v) in self.edges})

    def __pow__(self, power, modulo=None):
        return self if power == 1 else \
            ((self * self) ** int(power / 2)) * SPGraph(1, 2)

    def integer_mul(self, multiplier):
        return self.k2() if multiplier == 0 else \
            self if multiplier == 1 else \
            self + self if multiplier == 2 else \
            ((self * int((multiplier / 2))) * 2) + (self if multiplier % 2 == 1 else SPGraph(1, 1))

    def __rmul__(self, other):
        self.__mul__(other)

    # parallel composition or multiple
    def __mul__(self, other):
        if isinstance(other, int):
            return self.integer_mul(other)
        max_node = max(self.nodes)
        other = other.map_graph(lambda u: self.s if u == other.s else self.t if u == other.t else max_node + u - 1)
        s, t = self.s, other.t
        return SPGraph(s, t, set(self.edges).union(set(other.edges)))

    # series composition
    def __add__(self, other):
        max_node = max(self.nodes)
        other = other.map_graph(lambda u: self.t if u == other.s else max_node + u - 1)
        s, t = self.s, other.t
        return SPGraph(s, t, set(self.edges).union(set(other.edges)))


d = nx.Graph(list(((SPGraph.d1() * 2) ** 2).edges))
x = nx.linalg.graphmatrix.adjacency_matrix(d)
print(x.todense())
# np.savetxt("./graph_files/foo.csv", x.todense(), delimiter=',')
