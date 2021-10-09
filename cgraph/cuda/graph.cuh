//
// Created by yair on 08/08/2021.
//

#ifndef LEARNCPP_GRAPH_CUH
#define LEARNCPP_GRAPH_CUH

#include "../cgraph/graph.h"

struct Graph* dev_init_graph(int num_vertices);

__global__ void single_source_shortest_path(struct Graph* G, int* D);

struct Graph* copy_graph_to_device(struct Graph* G);

void print_device_graph(struct Graph* devG);

__global__ void get_first_neighbor(struct Graph* G, int* neighbors);

#endif //LEARNCPP_GRAPH_CUH
