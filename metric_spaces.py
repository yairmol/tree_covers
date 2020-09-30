from itertools import combinations
from typing import TypeVar, Callable, Generic, Set

import networkx as nx

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
Embedding = Callable[[T1], T2]

debug = False


class MetricSpace(Generic[T]):
    def __init__(self, points: Set[T], d: Callable[[T, T], float]):
        self.points = points
        self.d = d


def calculate_distortion(x_metric_space: MetricSpace[T1], y_metric_space: MetricSpace[T2], f: Embedding) -> float:
    distortion = 1.0
    for p1, p2 in combinations(x_metric_space.points, 2):
        f_p1 = f(p1)
        f_p2 = f(p2)
        d_x_p1_p2 = x_metric_space.d(p1, p2)
        d_y_p1_p2 = y_metric_space.d(f_p1, f_p2)
        curr_dist = d_y_p1_p2 / d_x_p1_p2
        if debug and curr_dist > 1.5:
            print(f"points ({p1}, {p2}) original distance is {d_x_p1_p2} but ({f_p1}, {f_p2}) distance is {d_y_p1_p2}")
            print(f"({p1}, {p2}) distortion: {curr_dist}")
        distortion = max(distortion, curr_dist)
    return distortion


def calculate_2tree_embedding_distortion(g, t1, t2):
    dist = calculate_distortion(MetricSpace(set(g.nodes), lambda u, v: len(nx.shortest_path(g, u, v)) - 1),
                                MetricSpace(set(t1.nodes), lambda u, v: min(len(nx.shortest_path(t1, u, v)),
                                                                            len(nx.shortest_path(t2, u, v))) - 1),
                                lambda x: x)
    return dist