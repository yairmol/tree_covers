// #include "utils/linked_list.h"
// #include "utils/vector.h"
#include "graph.h"
#include "graph_algorithms.h"
#include <stdio.h>
#include <stdlib.h>
// #include "series_parallel.h"
#include "embeddings.h"


// int main(int argc, char* argv[]){
//     int k = (argc > 1) ? atoi(argv[1]) : 7;
//     int start = (argc > 2) ? atoi(argv[2]) : 0;
//     struct Graph** graphs = two_tree_embedding(k);
//     printf("found two tree embedding\n");
//     struct Graph* G_1 = graphs[0], *T_1 = graphs[1], *T_2 = graphs[2];
//     int stop = (argc > 3) ? atoi(argv[3]) : G_1->num_vertices;
//     printf("%d %d %d\n", G_1->num_vertices, 64, G_1->num_vertices/64);
//     parallel_tree_embedding_distortion(G_1, T_1, T_2, start, stop);
//     return 0;
// }

// int main(){
//   // DecompTree * T = diamond_graph_decomp_tree(5);
//   SPGraph* SPG = series_composition(K2_sp_graph(), K2_sp_graph());
//   Graph* G = parallel_composition(SPG, SPG)->graph;
//   print_graph(G);
//   return 0;
// }

// int main(){
//     struct LinkedList ll1, ll2, ll3;
//     init_linked_list(&ll1); init_linked_list(&ll2);
//     for (int i = 0; i < 10; i++){
//         push_back(&ll1, i);
//     }
//     for (int i = 30; i < 40; i++){
//         push_back(&ll2, i);
//     }
//     ll3 = caten_linked_list(&ll1, &ll2);
//     print_linked_list(&ll1); printf("\n");
//     print_linked_list(&ll2); printf("\n");
//     print_linked_list(&ll3); printf("\n");
// }

// int main(){
//     Vector* v = vector();
//     for (int i = 0; i < 33; i++){
//         vector_insert(v, i);
//     }
//     print_vector(v);
//     printf("%d\n", vector_find(v, 3));
//     vector_remove(v, 3);
//     print_vector(v);
// }
int main(){
    struct Graph** graphs = two_tree_embedding(9);
    struct Graph* G = graphs[0];
    struct Graph* T1 = graphs[1];
    struct Graph* T2 = graphs[2];
    printf("%f\n", parallel_tree_embedding_distortion(G, T1, T2, 0, G->num_vertices));
}