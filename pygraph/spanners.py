from typing import Set
import networkx as nx
from metric_spaces import spanner_stretch, tree_cover_embedding_distortion, two_trees_embedding_distortion
from math import log2, floor
from networkx.utils import UnionFind

def low_stretch_spanner(G: nx.Graph, k: int):
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    for (u, v) in G.edges:
        try:
            if nx.shortest_path_length(H, u, v) > k - 1:
                H.add_edge(u, v)
        except Exception:
            H.add_edge(u, v)
    return H


def unordered_edges(edges: Set[tuple]):
    return {frozenset(e) for e in edges}


def simple_tree_cover(G: nx.Graph):
    k = 2 * floor(log2(len(G.nodes))) + 1
    H = low_stretch_spanner(G, k)
    print("spanner_stretch", spanner_stretch(G, H))
    print(H)
    T1 = nx.Graph(nx.dfs_tree(H))
    print(T1)
    T2 = nx.Graph()
    subtrees = UnionFind()
    for u, v in H.edges:
        if T1.has_edge(u, v):
            continue
        if subtrees[u] != subtrees[v]:
            T2.add_edge(u, v)
            subtrees.union(u, v)
    n = len(H.nodes)
    for u, v in H.edges:
        if len(T2.edges) >= n - 1:
            break
        if T2.has_edge(u, v):
            continue
        if subtrees[u] != subtrees[v]:
            T2.add_edge(u, v)
            subtrees.union(u, v)
    print("spanner tre cover dist", tree_cover_embedding_distortion(H, {T1, T2}))
    return T1, T2



def main():
    G = nx.gnp_random_graph(n=2048, p=0.01)
    if not nx.is_connected(G):
        print("G is not connected")
        return
    print(G)
    T1, T2 = simple_tree_cover(G)
    print(T1, nx.is_tree(T1), T2, nx.is_tree(T2))
    print(tree_cover_embedding_distortion(G, {T1, T2}))
    # k = 2 * log2(len(G.nodes)) + 1
    # print(k)
    # H = low_stretch_spanner(G, k)
    # print(H)
    # print(spanner_stretch(G, H))


if __name__ == "__main__":
    main()