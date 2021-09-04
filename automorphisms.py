import itertools as it
from typing import Union, Callable

import networkx as nx


def is_automorphism(g: nx.Graph, perm: Callable):
    x = list()
    for u, v in g.edges:
        x.append((perm(u), perm(v)))
        if not g.has_edge(perm(u), perm(v)):
            # print((u, v), (perm(u), perm(v)))
            return False
    return True


# g1 = nx.cycle_graph(4)
g1 = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 4), (1, 5), (2, 6)])
perm1 = [0, 2, 1, 3, 4, 6, 5]
perm2 = [0, 5, 6, 4, 3, 1, 2]
# print(is_automorphism(g1, lambda u: perm1[u]))
# print(is_automorphism(g1, lambda u: perm2[u]))
# print(is_automorphism(g1, lambda u: perm2[perm1[u]]))
for perm in it.permutations(g1.nodes, len(g1.nodes)):
    perm = [0] + list(perm)
    if is_automorphism(g1, lambda u: perm[u]):
        print(perm)
