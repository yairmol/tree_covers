#define GRAPH_VEC

#include "include/graph.h"
#include "../utils/include/vector.h"
#include <stdlib.h>
#include <stdio.h>

struct IGraph* init_graph(struct IGraph* G, int num_vertices){
  G->num_vertices = num_vertices;
  G->adj_list = (Vector*)calloc(num_vertices, sizeof(Vector));
  for (int i = 0; i < num_vertices; i++){
    vector_init(&G->adj_list[i]);
  }
  G->num_edges = 0;
  return G;
}

void free_igraph(struct IGraph* G){
  for (int i = 0; i < G->num_vertices; i++){
    free(G->adj_list[i].arr);
  }
  free(G->adj_list);
}

void add_edge(struct IGraph* G, int u, int v){
  vector_insert(&G->adj_list[u], v);
  vector_insert(&G->adj_list[v], u);
  G->num_edges++;
}

int remove_edge(struct IGraph* G, int u, int v){
  return vector_remove(&G->adj_list[u], v) 
         && vector_remove(&G->adj_list[v], u);
}

int has_edge(struct IGraph* G, int u, int v){
  return vector_find(&G->adj_list[u], v) != -1;
}

struct IGraph* copy_graph(struct IGraph* G){
  struct IGraph* G_c = (struct IGraph*)malloc(sizeof(struct IGraph));
  G_c->num_vertices = G->num_vertices;
  G_c->num_edges = G->num_edges;
  G_c->adj_list = (Vector*)calloc(G->num_vertices, sizeof(Vector));
  for (int i = 0; i < G->num_vertices; i++){
    vector_copy(&G->adj_list[i], &G_c->adj_list[i]);
  }
  return G_c;
}

struct EdgeGenerator* edges(struct IGraph* G){
  struct EdgeGenerator* EG = (struct EdgeGenerator*)malloc(sizeof(struct EdgeGenerator));
  EG->G = G;
  EG->current_u = 0;
  EG->next_v = 0;
  return EG;
}

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

struct IGraph* path_graph(struct IGraph* G, int n){
  init_graph(G, n);
  for (int u = 1; u < n; u++) {
    add_edge(G, u - 1, u);
  }
  return G;
}

struct IGraph* DiamondGraph(struct IGraph* G_k, int k){
  if (k < 1){
    return NULL;
  }
  if (k == 1) {
    init_graph(G_k, 4);
    add_edge(G_k, 0, 1); add_edge(G_k, 1, 2);
    add_edge(G_k, 2, 3); add_edge(G_k, 3, 0);
    return G_k;
  }
  struct IGraph G_k_minus_1; DiamondGraph(&G_k_minus_1, k - 1);
  init_graph(G_k, G_k_minus_1.num_vertices + (G_k_minus_1.num_edges * 2));
  struct EdgeGenerator* EG = edges(&G_k_minus_1);
  int u = G_k_minus_1.num_vertices;
  for (struct Edge e = next_edge(EG); e.u != 0 || e.v != 0; e = next_edge(EG)){
    add_edge(G_k, e.u, u); add_edge(G_k, e.u, u + 1);
    add_edge(G_k, e.v, u); add_edge(G_k, e.v, u + 1);
    u += 2;
  }
  free_igraph(&G_k_minus_1);
  return G_k;
}

void print_graph(struct IGraph* G) {
  for (int u = 0; u < G->num_vertices; u++){
    printf("%d: ", u);
    print_vector(&G->adj_list[u]);
  }
}
