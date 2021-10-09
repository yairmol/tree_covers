#include "graph_vec.h"
#include <stdio.h>
#include "graph_algorithms2.h"
#include "embeddings.h"

int main(){
    // struct Graph* G = DiamondGraph(2);
    struct Graph** graphs = two_tree_embedding(9);
    struct Graph* G = graphs[0];
    struct Graph* T1 = graphs[1];
    struct Graph* T2 = graphs[2];
    printf("%f\n", parallel_tree_embedding_distortion(G, T1, T2, 0, G->num_vertices));
    // print_graph(G);
    // int* D = single_source_shortest_path(G, 0);
    // for (int i = 0; i < G->num_vertices; i++){
    //     printf("%d: %d, ", i, D[i]);
    // }
    // printf("\n");
}