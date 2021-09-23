from datetime import datetime
from itertools import combinations
from time import time
from typing import TypeVar, Callable, Generic, Set, Dict, Any

import networkx as nx

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
Embedding = Callable[[T1], T2]

debug = True


class MetricSpace(Generic[T]):
    def __init__(self, points: Set[T], d: Callable[[T, T], float]):
        self.points = points
        self.d = d


def calculate_distortion(x_metric_space: MetricSpace[T1], y_metric_space: MetricSpace[T2], f: Embedding, threshold=1) -> float:
    distortion = 1.0
    for p1, p2 in combinations(x_metric_space.points, 2):
        f_p1 = f(p1)
        f_p2 = f(p2)
        d_x_p1_p2 = x_metric_space.d(p1, p2)
        d_y_p1_p2 = y_metric_space.d(f_p1, f_p2)
        curr_dist = d_y_p1_p2 / d_x_p1_p2
        if debug and curr_dist > threshold:
            print(f"points ({p1}, {p2}) original distance is {d_x_p1_p2} but ({f_p1}, {f_p2}) distance is {d_y_p1_p2}")
            print(f"({p1}, {p2}) distortion: {curr_dist}")
        distortion = max(distortion, curr_dist)
    return distortion


def _graph_distances(g: nx.Graph) -> Dict[Any, Dict[Any, int]]:
    paths_gen = nx.all_pairs_shortest_path(g)
    dists_dict = dict()
    i = 1
    for (u, paths_dict) in paths_gen:
        if i % 100 == 0:
            print(f"computed {i} vertices distances")
        dists_dict[u] = dict()
        for v in paths_dict:
            dists_dict[u][v] = len(paths_dict[v]) - 1
        i += 1
    return dists_dict


def calculate_2tree_embedding_distortion(g, t1, t2, threshold=1):
    print(datetime.now(), "finding all distances in g")
    g_dists = _graph_distances(g)
    print(datetime.now(), "done")
    return calc_2tree_embedding_distortion_with_known_distances(g_dists, t1, t2, threshold)


def calc_2tree_embedding_distortion_with_known_distances(g_dists: dict, t1, t2, threshold=1):
    print(datetime.now(), "finding all shortest paths t1")
    t1_dists = _graph_distances(t1)
    print(datetime.now(), "finding all shortest paths t2")
    t2_dists = _graph_distances(t2)
    print("done")
    dist = calculate_distortion(MetricSpace(set(t1.nodes), lambda u, v: g_dists[u][v]),
                                MetricSpace(set(t1.nodes), lambda u, v: min(t1_dists[u][v] if v in t1_dists[u] else t1_dists[v][u],
                                                                            t2_dists[u][v] if v in t2_dists[u] else t1_dists[v][u])),
                                lambda x: x, threshold)
    return dist


def calc_2trees_embedding_distortion_low_memory(g, t1, t2, threshold=1):
    max_distortion = 1
    i = 1
    for u in g:
        if i % 100 == 0:
            print(datetime.now(), f"finished {i} vertices")
        u_g_paths_dict = nx.single_source_shortest_path_length(g, u)
        u_t1_paths_dict = nx.single_source_shortest_path_length(t1, u)
        u_t2_paths_dict = nx.single_source_shortest_path_length(t2, u)
        for v in u_g_paths_dict:
            if u >= v:
                continue
            g_dist = len(u_g_paths_dict[v]) - 1
            t1_dist = len(u_t1_paths_dict[v]) - 1
            t2_dist = len(u_t2_paths_dict[v]) - 1
            distortion = min(t1_dist, t2_dist) / g_dist
            if distortion > threshold:
                print(f"{u}, {v} real distance is {g_dist} but in t1, t2 is {min(t1_dist, t2_dist)}. "
                      f"distortion: {distortion}")
            max_distortion = max(max_distortion, distortion)
        i += 1
    return max_distortion


def _generate_dists(g_paths_gen, batch_size) -> Dict[Any, Dict[Any, int]]:
    dists_dict = dict()
    i = 1
    for (u, paths_dict) in g_paths_gen:
        dists_dict[u] = paths_dict
        if i >= batch_size:
            break
        i += 1
    return dists_dict


def calc_2trees_embedding_distortion_medium_memory(
        g: nx.Graph, t1: nx.Graph, t2: nx.Graph,
        threshold=1, batch_size=1400
):
    g_all_paths_gen = nx.all_pairs_shortest_path_length(g)
    t1_all_paths_gen = nx.all_pairs_shortest_path_length(t1)
    t2_all_paths_gen = nx.all_pairs_shortest_path_length(t2)
    max_distortion = 1
    for i in range(0, len(g.nodes), batch_size):
        g_dists = _generate_dists(g_all_paths_gen, batch_size)
        t1_dists = _generate_dists(t1_all_paths_gen, batch_size)
        t2_dists = _generate_dists(t2_all_paths_gen, batch_size)
        for u in g_dists:
            for v in g_dists[u]:
                g_dist = g_dists[u][v]
                t1_dist = t1_dists[u][v]
                t2_dist = t2_dists[u][v]
                distortion = min(t1_dist, t2_dist) / g_dist
                if distortion > threshold:
                    print(f"{u}, {v} real distance is {g_dist} but in t1, t2 is {min(t1_dist, t2_dist)}. "
                          f"distortion: {distortion}")
                max_distortion = max(max_distortion, distortion)
    return max_distortion


# def calculate_distortion_to_ancestor(g: nx.Graph, t1: nx.Graph, t2: nx.Graph, e_ancestor, successors=None):
#     u, v = e_ancestor
#     dists_g = {
#         u: nx.single_source_shortest_path_length(g, u),
#         v: nx.single_source_shortest_path_length(g, v)
#     }
#     dists_t1 = {
#         u: nx.single_source_shortest_path_length(t1, u),
#         v: nx.single_source_shortest_path_length(t1, v)
#     }
#     dists_t2 = {
#         u: nx.single_source_shortest_path_length(t2, u),
#         v: nx.single_source_shortest_path_length(t2, v)
#     }
#     if not successors:
#         successors = g.nodes
#     for successor in successors:


