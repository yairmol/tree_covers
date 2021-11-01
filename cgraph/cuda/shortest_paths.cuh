#ifndef CU_SHORTEST_PATHS_H
#define CU_SHORTEST_PATHS_H

#include "graph.cuh"

/**
 * calculates distances from s to all other vertices in G
 * @param G A Graph
 * @param s A vertex in G from which to calculate all distances
 * @return an array of distances d, where d[i] = d_G(s, i)
 */
__global__
void single_source_shortest_path(struct Graph* G, int offset, u_int8_t** D);

/**
 * calculates distances for all u, v in V(G)
 * @param G A Graph
 * @return a distance matrix D.
 */
__host__
void all_pairs_shortest_path(struct Graph* G, u_int8_t** D);

#endif //CU_SHORTEST_PATHS_H
