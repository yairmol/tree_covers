#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "../graph/include/graph.h"
#include <stdlib.h>

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

/**
 * @brief returns the stretch of the tree cover Ts with respect to G
 * 
 * @param G a graph
 * @param Ts an array of trees which is a tree cover of G
 * @param length size of the tree cover Ts
 * @return float the stretch of Ts with respect to G
 */
float tree_cover_embedding_distortion(struct IGraph* G, struct IGraph* Ts, size_t length);

typedef float (*metric_t)(int, int);

/**
 * @brief returns the embedding distortion of d2 with respect to d1 and a set X
 * assume the embedding function itself is the identity
 * 
 * @param X an array which represents the set of the metric space (X, d1)
 * @param size the size of the array X
 * @param d1 original metric on X
 * @param d2 embedded metric on X
 * @return the embedding distortion
 */
float embedding_distortion(int* X, size_t size, metric_t d1, metric_t d2);
#endif // LEARNCPP_EMBEDDING_H