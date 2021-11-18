
#ifndef LEARNCPP_GRAPHVEC_H
#define LEARNCPP_GRAPHVEC_H

#include "../../utils/include/vector.h"

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
struct Graph* init_graph(int num_vertices);

/**
 * Free the memory of an allocated graph
 */
void free_graph(struct Graph* G);

/**
 * Add an edge to G
 * the function doesn't check that u and v are within the range of G->num_vertices
 * @param G a graph
 * @param u one vertex in the new edge
 * @param v second vertex in the new edge
 */
void add_edge(struct Graph* G, int u, int v);

/**
 * removes an edge from G
 * return 1 if it exists and 0 otherwise
 * @param G a graph
 * @param u one vertex in the new edge
 * @param v second vertex in the new edge
 */
int remove_edge(struct Graph* G, int u, int v);

// checks if the edge {u, v} is in G
int has_edge(struct Graph* G, int u, int v);

/**
 * creates a deep copy of G
 */
struct Graph* copy_graph(struct Graph* G);

/**
 * A struct for edge iteration
 * given a graph one can iterate over the edges using this struct and the next function
 */
struct EdgeGenerator{
    struct Graph* G;
    int current_u;
    int next_v;
};

struct EdgeGenerator* edges(struct Graph* G);

struct Edge{
    int u;
    int v;
};

/**
 * A next function for the edge iterator
 */
struct Edge next_edge(struct EdgeGenerator* EG);

struct Graph* path_graph(int n);

struct Graph* DiamondGraph(int k);

void print_graph(struct Graph* G);
#endif //LEARNCPP_GRAPH_H
