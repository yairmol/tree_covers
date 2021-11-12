from itertools import combinations
from typing import List, Optional, Union, Any, Tuple, Callable
import networkx as nx
import numpy as np
from enum import Enum, auto


Vertex = int
Edge = Tuple[Vertex, Vertex]


class SPTreeLabel(Enum):
    K2 = auto()
    SERIES = auto()
    PARALLEL = auto()


class SPGraph(nx.Graph):
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
        # TODO: fix for odd power
        return (self if power == 1 else ((self * self) ** int(power / 2)))

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
        max_node = max(self.nodes) + 1

        def mapping(u):
            if u < other.s:
                return max_node + u
            if u == other.s:
                return self.s
            if other.s < u < other.t:
                return max_node + u - 1
            if u == other.t:
                return self.t
            if u > other.t:
                return u + max_node - 2
        
        # lambda u: self.s if u == other.s else self.t if u == other.t else max_node + u - 1
        other = other.map_graph(mapping)
        s, t = self.s, other.t
        return SPGraph(s, t, set(self.edges).union(set(other.edges)))

    # series composition
    def __add__(self, other):
        max_node = max(self.nodes) + 1
        s2 = other.s
        other = other.map_graph(lambda u: self.t if u == s2 else max_node + u if u < s2 else max_node + u - 1)
        s, t = self.s, other.t
        return SPGraph(s, t, set(self.edges).union(set(other.edges)))
    
    @staticmethod
    def k2():
        k2 = SPGraph(s=0, t=1)
        k2.add_edge(0, 1)
        return k2

    @staticmethod
    def d1():
        return SPGraph(s=1, t=3, incoming_graph_data={(1, 2), (2, 3), (3, 4), (4, 1)})
    
    @staticmethod
    def dk(k: int):
        if k < 0:
            raise ValueError("k must be non-negative")
        return SPGraph.k2() if k == 0 else (SPGraph.dk(k - 1) * 2) ** 2
    
    @staticmethod
    def construct_from_labeled_tree(T: nx.DiGraph, r):
        """
        construct a series parallel graph from a rooted labeled tree T.
        a rooted labeled tree T, is a binary tree rooted at r, whose leaves all represents K2 graphs
        and all other inner nodes represent either a parallel or a series composition
        """
        if T.out_degree(r) == 0:
            assert T.nodes[r]['label'] == SPTreeLabel.K2
            return SPGraph.k2()
        assert T.out_degree(r) == 2, "not a binary tree"
        f = SPGraph.construct_from_labeled_tree
        SPG1, SPG2 = tuple(f(T, ri) for ri in T.succ[r])
        if len(SPG1.nodes) == 2 and SPG2.has_edge(SPG2.s, SPG2.t):
            return SPG1 + SPG2
        return SPG1 + SPG2 if T.nodes[r]['label'] == SPTreeLabel.SERIES else SPG1 * SPG2

    @staticmethod
    def random(m: int):
        """
        returns a random series parallel graph on m edges
        """
        def random_binary_tree(n: int):
            """
            generates a random binary tree with n leaves
            returns it and the root
            """
            leaves = list(range(n))
            T = nx.DiGraph()
            T.add_nodes_from(leaves)
            counter = n
            while len(leaves) != 1:
                n1, n2 = np.random.choice(leaves, 2, replace=False)
                i1 = leaves.index(n1)
                leaves.remove(n1), leaves.remove(n2)
                T.add_edge(counter, n1)
                T.add_edge(counter, n2)
                leaves.insert(i1, counter)
                counter += 1
            return T, counter - 1
        
        def random_labeling(T: nx.DiGraph):
            for n, data in T.nodes(data=True):
                if T.out_degree(n) == 0:
                    data['label'] = SPTreeLabel.K2
                    continue
                data['label'] = (
                    SPTreeLabel.PARALLEL if np.random.randint(2)
                    else SPTreeLabel.SERIES
                )
        
        T, r = random_binary_tree(m)
        random_labeling(T)
        return SPGraph.construct_from_labeled_tree(T, r)
    
    @staticmethod
    def is_sp_graph(G: nx.Graph):
        """
        return True if G is a Series-Parallel Graph

        A Graph 𝐺 = (𝑉, 𝐸) is Series-Parallel 
        ⇔ 
        ∃ 𝑠,𝑡 ∈ 𝑉 such that 𝐾₂ can be achieved from 𝐺 by
        iteratively removing any vertex 𝑣 with degree 2 and
        replacing it with a with a single edge between v's neighbors
        (and removing parallel edges)
        """
        for u, v in combinations(G.nodes, 2):
            G_copy = nx.Graph(G)
            success = True
            while len(G_copy.nodes) != 2:
                deg2_vertices = [w for w in G_copy.nodes if G_copy.degree(w) == 2 and w not in {u, v}] #not {u, v}.intersection(set(G_copy[w]) | {w})]
                if not deg2_vertices:
                    success = False
                    break
                u = deg2_vertices[0]
                u_neighbors = list(G_copy[u])
                G_copy.remove_node(u)
                G_copy.add_edge(u_neighbors[0], u_neighbors[1])
            if success:
                return True
        return False


    # def is_sp_graph(G: nx.Graph):
    #     """
    #     A graph G is series-parallel if it can be built using a sequence of series and parallel
    #     compositions of K2 graphs. this is equivalent to a being 1-seperated or 2-seperated into
    #     2 series-parallel graph
    #     """
    #     if len(G.nodes) < 2:
    #         print("problem")
    #         return False
            
    #     if len(G.nodes) == 2 and G.has_edge(*G.nodes):
    #         return True
        
    #     V = set(G.nodes)
        
    #     def is_sp_separable(seperator: set):
    #         Gtag = nx.induced_subgraph(G, V - seperator)
    #         CCs = list(nx.connected_components(Gtag))
    #         if len(CCs) > 2:
    #             print("many CCs", len(CCs))
    #         other_CCs = lambda CC: [CC1 for CC1 in CCs if CC1 != CC]
    #         # print(f"{seperator} seperating", len(CCs) > 1)
    #         ret = (
    #             len(CCs) > 1 and
    #             all(is_sp_graph(nx.induced_subgraph(G, V.difference(*other_CCs(CC)))) for CC in CCs)
    #         )
    #         # print(seperator, ret)
    #         return ret

    #     return (
    #         any(is_sp_separable({u}) for u in G.nodes) or
    #         any(is_sp_separable({u, v}) for u, v in combinations(G.nodes, 2))
    #     )


def main():
    # G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 6), (6, 2)])
    G = nx.Graph(SPGraph.random(2000))
    # G = nx.complete_graph(4)
    print("graph build")
    # G = (SPGraph.k2() + SPGraph.k2()) * (SPGraph.k2() + SPGraph.k2()) + (SPGraph.k2() * 3)
    # print(G.nodes)
    # print(G.edges)
    # print(G.edges)
    # print(is_sp_graph(nx.Graph(SPGraph.k2() * 4)))
    print(SPGraph.is_sp_graph(nx.Graph(G)))


if __name__ == "__main__":
    main()