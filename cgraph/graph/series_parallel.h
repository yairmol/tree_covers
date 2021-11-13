#ifndef LEARNCPP_SERIES_PARALLEL_H
#define LEARNCPP_SERIES_PARALLEL_H

#include "graph.h"
#include "../utils/queue.h"
#include <stdlib.h>
#include <stdio.h>

typedef struct Graph Graph;
typedef struct Edge Edge;

/**
 *  G = (V, E, s, t) is a serial parallel graph if it is 
 * a 2-terminal graph that is the result of serial and parallel compositions
 * starting from K2
 */
typedef struct SPGraph {
    Graph* graph;
    int s;
    int t;
} SPGraph;


enum Composition {
    K2,
    SERIES_COMPOSITION,
    PARALLEL_COMPOSITION
};

/**
 * A Decomposition Tree is a binary tree such that every
 * node in the tree represents a composition operation (series or parallel)
 * and the leaves are K2 graphs
 * for convinience every node also holds the appropriate sp-graph
 */
typedef struct DecompTree {
    enum Composition composition;
    struct DecompTree* left;
    struct DecompTree* right;
    struct SPGraph* G;
} DecompTree;

// build and return a decomposition tree of the diamond graph
DecompTree* diamond_graph_decomp_tree(int k);


char* composition_to_string(enum Composition comp);


int height(DecompTree* T);


void print_decomp_tree(DecompTree* T);

// allocates and initializes an sp-graph
SPGraph* init_sp_graph(int num_vertices, int s, int t);

// allocates and initializes a K2 sp-graph
SPGraph* K2_sp_graph();


SPGraph* parallel_composition(SPGraph* G1, SPGraph* G2);


SPGraph* series_composition(SPGraph* G1, SPGraph* G2);


#endif // LEARNCPP_SERIES_PARALLEL_H