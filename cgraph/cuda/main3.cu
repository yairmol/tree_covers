// #include "graph.cuh"

// struct Graph** two_tree_embedding(int k){
//   if (k < 1){
//     return NULL;
//   }
//   if (k == 1) {
//     struct Graph* G_1 = init_graph(4), *T_1 = init_graph(4), *T_2 = init_graph(4);
//     add_edge(G_1, 0, 1); add_edge(T_2, 0, 1);
//     add_edge(G_1, 1, 2); add_edge(T_1, 1, 2);
//     add_edge(G_1, 2, 3); add_edge(T_1, 2, 3); add_edge(T_2, 2, 3);
//     add_edge(G_1, 3, 0); add_edge(T_1, 3, 0); add_edge(T_2, 3, 0);
//     struct Graph** ret = (struct Graph**)malloc(3 * sizeof(struct Graph*));
//     ret[0] = G_1;
//     ret[1] = T_1;
//     ret[2] = T_2;
//     return ret;
//   }
//   struct Graph** graphs = two_tree_embedding(k - 1);
//   struct Graph* G_k_minus_1 = graphs[0], *T_k_minus_1_1 = graphs[1], *T_k_minus_1_2 = graphs[2];
//   int k_num_vertices = G_k_minus_1->num_vertices + (G_k_minus_1->num_edges * 2);
//   struct Graph* G_k = init_graph(k_num_vertices), *T_1 = init_graph(k_num_vertices), *T_2 = init_graph(k_num_vertices);
//   struct EdgeGenerator* EG = edges(G_k_minus_1);
//   int u = G_k_minus_1->num_vertices;
//   for (struct Edge e = next_edge(EG); e.u != 0 || e.v != 0; e = next_edge(EG)){
//     if (e.v < e.u){
//       int tmp = e.v;
//       e.v = e.u;
//       e.u = tmp;
//     }
//     add_edge(G_k, e.u, u); add_edge(G_k, e.u, u + 1);
//     add_edge(G_k, e.v, u); add_edge(G_k, e.v, u + 1);
//     int t_1_has_edge = has_edge(T_k_minus_1_1, e.u, e.v), t_2_has_edge = has_edge(T_k_minus_1_2, e.u, e.v);
//     if (!t_1_has_edge && !t_2_has_edge){
//       printf("----- %d: both t1 and t2 doesn't have an edge {%d, %d}\n", k, e.u, e.v);
//     }
//     if (t_1_has_edge && t_2_has_edge){
//       add_edge(T_1, e.u, u + 1); add_edge(T_1, e.v, u); add_edge(T_1, e.v, u + 1);
//       add_edge(T_2, e.u, u); add_edge(T_2, e.u, u + 1); add_edge(T_2, e.v, u);
//     }
//     else if (t_1_has_edge){
//       add_edge(T_1, e.u, u + 1); add_edge(T_1, e.v, u); add_edge(T_1, e.v, u + 1);
//       add_edge(T_2, e.u, u); add_edge(T_2, e.u, u + 1);
//     }
//     else if (t_2_has_edge){
//       add_edge(T_1, e.u, u); add_edge(T_1, e.u, u + 1);
//       add_edge(T_2, e.u, u); add_edge(T_2, e.v, u); add_edge(T_2, e.v, u + 1);
//     }
//     u += 2;
//   }
//   free_graph(G_k_minus_1); free_graph(T_k_minus_1_1); free_graph(T_k_minus_1_2);
//   free(graphs);
//   struct Graph** ret = (struct Graph**)malloc(3 * sizeof(struct Graph*));
//   ret[0] = G_k;
//   ret[1] = T_1;
//   ret[2] = T_2;
//   return ret;
// }


// int main(int argc, char** argv){
//     if (argc < 2) {
//         printf("Usage: gpugraph k [start] [stop]");
//     }
//     int k = atoi(argv[1]);
//     printf("building diamond graph %d\n", k);
//     struct Graph** graphs = two_tree_embedding(k);
//     struct Graph* G = graphs[0];
//     struct Graph* T1 = graphs[1];
//     struct Graph* T2 = graphs[2];
//     printf("num edges: %d, num vertices: %d\n", G->num_edges, G->num_vertices);
//     printf("T1 is tree %d\n", is_tree(T1));
//     printf("T2 is tree %d\n", is_tree(T2));
//     int start = (argc > 2) ? atoi(argv[2]) : 0;
//     int stop = (argc > 3) ? atoi(argv[3]) : G->num_vertices;
//     printf("running embedding distortion from %d to %d\n", start, stop);
//     printf("%f\n", parallel_tree_embedding_distortion(G, T1, T2, start, stop));
// }

// #include "vector.cuh"
// #include "graph.cuh"
#include "vector.cuh"
#include "graph.cuh"
#include "tree_covers.cuh"
#include "shortest_paths.cuh"
#include <stdio.h>

int main(){
    struct Graph graphs[3];
    two_tree_embedding(graphs, 3);
    printf("\nG:\n");
    print_graph(&graphs[0]);
    printf("\nT1:\n");
    print_graph(&graphs[1]);
    printf("\nT2:\n");
    print_graph(&graphs[2]);
    u_int8_t** D = (u_int8_t**)malloc(graphs[0].num_vertices * graphs[0].num_vertices * sizeof(u_int8_t));
    all_pairs_shortest_path(&graphs[0], D);
    for (int i = 0; i < graphs[0].num_vertices; i++){
        printf("%u ", D[0][i]);
    }
}
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

// 13, 4, 5, 6, 7, 8, 3, 11