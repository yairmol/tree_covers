#ifndef LEARNCPP_GRAPH_ALGORITHMS_H
#define LEARNCPP_GRAPH_ALGORITHMS_H

#include "graph_vec.h"

/**
 * calculates distances from s to all other vertices in G
 * @param G A Graph
 * @param s A vertex in G from which to calculate all distances
 * @return an array of distances d, where d[i] = d_G(s, i)
 */
int* single_source_shortest_path(struct Graph* G, int s);

/**
 * calculates distances for all u, v in V(G)
 * @param G A Graph
 * @return a distance matrix D.
 */
int** all_pairs_shortest_path(struct Graph* G);

struct DistanceGenerator {
    struct Graph* G;
    int current;
    int max_node;
};

int* next(struct DistanceGenerator* DG);

struct DistanceGenerator* init_distance_generator(struct Graph* G, int start, int stop);

/**
 * creates a generator for all distances in G
 * every item in the generator is an array of distances
 */
struct DistanceGenerator* all_pairs_shortest_paths_length_generator(struct Graph* G);

/**
 * @brief returns a graph which is the bfs tree of G rooted at source
 * 
 * @param G a graph
 * @param source the source from which we run bfs
 * @return a tree
 */
struct Graph* bfs_tree(struct Graph* G, int source);

/**
 * @brief returns a graph which is a dfs tree (or forest) of G
 * 
 * @param G a graph
 * @return a Tree (or forest) 
 */
struct Graph* dfs_tree(struct Graph* G);

/**
 * @brief find the connected componenets of G
 * 
 * @param G a graph
 * @return an array of vectors where each vector is a connected component
 */
Vector* connected_components(struct Graph* G);

/**
 * @brief returns a graph which is the induced subgraph of G on U 
 * 
 * @param G a graph
 * @param U a set of vertices of G (U ⊆ V) 
 * @return an induced subgraph
 */
struct Graph* induced_subgraph(struct Graph* G, int* U);

#endif //LEARNCPP_GRAPH_ALGORITHMS_H
