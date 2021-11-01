#include "graph.cuh"
#include "vector.cuh"
#include <stdlib.h>
#include <stdio.h>


__device__ __host__
void mmalloc(void** p, size_t size){
  #ifdef __CUDA_ARCH__
  *p = malloc(size);
  #else
  cudaMallocManaged(p, size);
  #endif

}

__device__ __host__
void mfree(void* p){
  #ifdef __CUDA_ARCH__
  free(p);
  #else
  cudaFree(p);
  #endif
}

__host__ __device__
struct Graph* init_graph(struct Graph* G, int num_vertices){
  G->num_vertices = num_vertices;
  mmalloc((void**)&G->adj_list, num_vertices * sizeof(Vector));
  for (int i = 0; i < num_vertices; i++){
    vector_init(&G->adj_list[i]);
  }
  G->num_edges = 0;
  return G;
}

__host__ __device__
void free_graph(struct Graph* G){
  for (int i = 0; i < G->num_vertices; i++){
    vector_free(&G->adj_list[i]);
  }
  mfree(G->adj_list);
}

__host__ __device__
void add_edge(struct Graph* G, int u, int v){
  vector_insert(&G->adj_list[u], v);
  vector_insert(&G->adj_list[v], u);
  G->num_edges++;
}

__host__ __device__
int remove_edge(struct Graph* G, int u, int v){
  return vector_remove(&G->adj_list[u], v) 
         && vector_remove(&G->adj_list[v], u);
}

__host__ __device__
int has_edge(struct Graph* G, int u, int v){
  return vector_find(&G->adj_list[u], v) != -1;
}

__host__ __device__
struct EdgeGenerator* edges(struct EdgeGenerator* EG, struct Graph* G){
  EG->G = G;
  EG->current_u = 0;
  EG->next_v = 0;
  return EG;
}

__host__ __device__
struct Edge next_edge(struct EdgeGenerator* EG){
start:
  if (EG->current_u >= EG->G->num_vertices){
    return (struct Edge){0, 0};
  }
  if(EG->next_v >= EG->G->adj_list[EG->current_u].cur){
    EG->current_u++;
    EG->next_v = 0;
    goto start;
  }
  if(EG->G->adj_list[EG->current_u].arr[EG->next_v] < EG->current_u){
    EG->next_v++;
    goto start;
  }
  struct Edge e = {EG->current_u, EG->G->adj_list[EG->current_u].arr[EG->next_v]};
  EG->next_v++;
  return e;
}

__host__ __device__
struct Graph* path_graph(struct Graph* G, int n){
  init_graph(G, n);
  for (int u = 1; u < n; u++) {
    add_edge(G, u - 1, u);
  }
  return G;
}

__host__ __device__
struct Graph* DiamondGraph(struct Graph* Gk, int k){
  if (k < 1){
    return NULL;
  }
  if (k == 1) {
    init_graph(Gk, 4);
    add_edge(Gk, 0, 1); add_edge(Gk, 1, 2);
    add_edge(Gk, 2, 3); add_edge(Gk, 3, 0);
    return Gk;
  }
  struct Graph G_k_minus_1;
  DiamondGraph(&G_k_minus_1, k - 1);
  init_graph(Gk, G_k_minus_1.num_vertices + (G_k_minus_1.num_edges * 2));
  struct EdgeGenerator EG;
  edges(&EG, &G_k_minus_1);
  int u = G_k_minus_1.num_vertices;
  for (struct Edge e = next_edge(&EG); e.u != 0 || e.v != 0; e = next_edge(&EG)){
    add_edge(Gk, e.u, u); add_edge(Gk, e.u, u + 1);
    add_edge(Gk, e.v, u); add_edge(Gk, e.v, u + 1);
    u += 2;
  }
  free_graph(&G_k_minus_1);
  return Gk;
}

__host__
void print_graph(struct Graph* G) {
  for (int u = 0; u < G->num_vertices; u++){
    printf("%d: ", u);
    print_vector(&G->adj_list[u]);
  }
}
