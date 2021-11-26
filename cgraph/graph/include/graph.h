
#ifndef CGRAPH_H
#define CGRAPH_H

#ifdef GRAPH_VEC
#include "../../utils/include/vector.h"
typedef Vector adj_list_t;
#else
#include "../../utils/include/linked_list.h"
typedef struct LinkedList adj_list_t;
#endif

struct IGraph {
    int num_vertices;
    int num_edges;
    adj_list_t* adj_list;
};

/**
 * Allocate and intialize a graph with num_vertices vertices {0, ... , n - 1}
 * @param num_vertices number of vertices in the created graph+
 * @return a graph with num_vertices vertices
 */
struct IGraph* init_graph(struct IGraph* G, int num_vertices);

/**
 * Free the memory of an allocated graph
 */
void free_igraph(struct IGraph* G);

/**
 * Add an edge to G
 * the function doesn't check that u and v are within the range of G->num_vertices
 * @param G a graph
 * @param u one vertex in the new edge
 * @param v second vertex in the new edge
 */
void add_edge(struct IGraph* G, int u, int v);

/**
 * removes an edge from G
 * return 1 if it exists and 0 otherwise
 * @param G a graph
 * @param u one vertex in the new edge
 * @param v second vertex in the new edge
 */
int remove_edge(struct IGraph* G, int u, int v);

// checks if the edge {u, v} is in G
int has_edge(struct IGraph* G, int u, int v);

/**
 * creates a deep copy of G
 */
struct IGraph* copy_graph(struct IGraph* G);

/**
 * A struct for edge iteration
 * given a graph one can iterate over the edges using this struct and the next function
 */
struct EdgeGenerator{
    struct IGraph* G;
    int current_u;
    #ifdef GRAPH_VEC
    int next_v;
    #else
    struct Link* next_v;
    #endif
};

struct EdgeGenerator* edges(struct IGraph* G);

struct Edge{
    int u;
    int v;
};

/**
 * A next function for the edge iterator
 */
struct Edge next_edge(struct EdgeGenerator* EG);

struct IGraph* path_graph(struct IGraph* G, int n);

struct IGraph* DiamondGraph(struct IGraph* G, int k);

void print_graph(struct IGraph* G);

#endif //CGRAPH_H
