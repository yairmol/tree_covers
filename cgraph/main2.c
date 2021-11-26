#include "graph/include/graph.h"
#include <stdio.h>
#include <stdlib.h>
#include "graph/include/graph_algorithms.h"
#include "embeddings/embeddings.h"


int is_tree(struct IGraph* T){
    int* d_T = single_source_shortest_path(T, 0);
    printf("num edges: %d, num vertices: %d\n", T->num_edges, T->num_vertices);
    int ret = 1;
    for (size_t i = 1; i < T->num_vertices; i++) {
        ret = ret && (d_T[i] > 0);
    }
    return ret && (T->num_edges == T->num_vertices - 1);
    
}


int main(int argc, char** argv){
    if (argc < 2) {
        printf("Usage: graph2 k [start] [stop]");
    }
    int k = atoi(argv[1]);
    printf("building diamond graph %d\n", k);
    struct IGraph G, T1, T2;
    two_tree_embedding(&G, &T1, &T2, k);
    printf("num edges: %d, num vertices: %d\n", G.num_edges, G.num_vertices);
    printf("T1 is tree %d\n", is_tree(&T1));
    printf("T2 is tree %d\n", is_tree(&T2));
    int start = (argc > 2) ? atoi(argv[2]) : 0;
    int stop = (argc > 3) ? atoi(argv[3]) : G.num_vertices;
    printf("running embedding distortion from %d to %d\n", start, stop);
    printf("%f\n", parallel_tree_embedding_distortion(&G, &T1, &T2, start, stop));
}

// int main(){
//     struct IGraph G;
//     path_graph(&G, 10);
//     // print_graph(&G);
//     int* d = single_source_shortest_path(&G, 0);
//     for (int i = 0; i < G.num_vertices; i++)
//     {
//         printf("d(0, %d) = %d\n", i, d[i]);
//     }
    
// }