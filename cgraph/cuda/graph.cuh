
#ifndef LEARNCPP_GRAPHVEC_H
#define LEARNCPP_GRAPHVEC_H

#include "vector.cuh"

struct Graph {
    int num_vertices;
    int num_edges;
    Vector* adj_list;
};

/**
 * Allocate and intialize a graph with num_vertices vertices {0, ... , n - 1}
 * @param num_vertices number of vertices in the created graph+
 * @return a graph with num_vertices vertices
 */
__device__ __host__
struct Graph* init_graph(struct Graph* G, int num_vertices);

/**
 * Free the memory of an allocated graph
 */
__device__ __host__
void free_graph(struct Graph* G);

/**
 * Add an edge to G
 * the function doesn't check that u and v are within the range of G->num_vertices
 * @param G a graph
 * @param u one vertex in the new edge
 * @param v second vertex in the new edge
 */
__device__ __host__
void add_edge(struct Graph* G, int u, int v);

/**
 * removes an edge from G
 * return 1 if it exists and 0 otherwise
 * @param G a graph
 * @param u one vertex in the new edge
 * @param v second vertex in the new edge
 */
__device__ __host__
int remove_edge(struct Graph* G, int u, int v);

// checks if the edge {u, v} is in G
__device__ __host__
int has_edge(struct Graph* G, int u, int v);

/**
 * A struct for edge iteration
 * given a graph one can iterate over the edges using this struct and the next function
 */
struct EdgeGenerator{
    struct Graph* G;
    int current_u;
    int next_v;
};

__device__ __host__
struct EdgeGenerator* edges(struct EdgeGenerator* EG, struct Graph* G);

struct Edge{
    int u;
    int v;
};

/**
 * A next function for the edge iterator
 */
__device__ __host__
struct Edge next_edge(struct EdgeGenerator* EG);

__device__ __host__
struct Graph* path_graph(struct Graph* G, int n);

__device__ __host__
struct Graph* DiamondGraph(struct Graph* Gk, int k);

__host__
void print_graph(struct Graph* G);

#endif //LEARNCPP_GRAPH_H
