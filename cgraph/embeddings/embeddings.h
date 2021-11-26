#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "../graph/include/graph.h"

/**
 * @param G a graph
 * @param T_1 a spanning tree of G
 * @param T_2 a spanning tree of G
 * @return the embedding distortion which is defined as max{min(d_T1(u, v), d_T2(u, v)) / d_G(u, v) | u, v in V(G)}
 */
float tree_embedding_distortion(struct IGraph* G, struct IGraph* T_1, struct IGraph* T_2);

/**
 * calculate the embedding distortion in parallel.
 * @see tree_embedding_distortion() for more info
 */
float parallel_tree_embedding_distortion(struct IGraph* G, struct IGraph* T_1, struct IGraph* T_2, int all_start, int all_stop);

#endif // LEARNCPP_EMBEDDING_H