#ifndef LEARNCPP_GRAPH_ALGORITHMS_H
#define LEARNCPP_GRAPH_ALGORITHMS_H

#include "graph.h"

/**
 * calculates distances from s to all other vertices in G
 * @param G A Graph
 * @param s A vertex in G from which to calculate all distances
 * @return an array of distances d, where d[i] = d_G(s, i)
 */
int* single_source_shortest_path(struct IGraph* G, int s);

/**
 * calculates distances for all u, v in V(G)
 * @param G A Graph
 * @return a distance matrix D.
 */
int** all_pairs_shortest_path(struct IGraph* G);

struct DistanceGenerator {
    struct IGraph* G;
    int current;
    int max_node;
};

int* next(struct DistanceGenerator* DG);

struct DistanceGenerator* init_distance_generator(struct IGraph* G, int start, int stop);

/**
 * creates a generator for all distances in G
 * every item in the generator is an array of distances
 */
struct DistanceGenerator* all_pairs_shortest_paths_length_generator(struct IGraph* G);

int two_tree_embedding(struct IGraph* G_k, struct IGraph* T_1, struct IGraph* T_2, int k);

#endif //LEARNCPP_GRAPH_ALGORITHMS_H
