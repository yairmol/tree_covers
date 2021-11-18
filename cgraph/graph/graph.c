#include "include/graph.h"
#include "../utils/include/linked_list.h"
#include <stdlib.h>
#include <stdio.h>

struct Graph* init_graph(int num_vertices){
  struct Graph* G = (struct Graph*)malloc(sizeof(struct Graph));
  G->num_vertices = num_vertices;
  G->adj_list = (struct LinkedList*)calloc(num_vertices, sizeof(struct LinkedList));
  G->num_edges = 0;
  return G;
}

void free_graph(struct Graph* G){
  for (int i = 0; i < G->num_vertices; i++){
    free_link(G->adj_list[i].head);
  }
  free(G->adj_list);
  free(G);
}

void add_edge(struct Graph* G, int u, int v){
  insert(&G->adj_list[u], v);
  insert(&G->adj_list[v], u);
  G->num_edges++;
}

int remove_edge(struct Graph* G, int u, int v){
  return linked_list_remove(&G->adj_list[u], v) 
         && linked_list_remove(&G->adj_list[v], u);
}

int has_edge(struct Graph* G, int u, int v){
  struct Link* ptr = G->adj_list[u].head;
  while (ptr != NULL){
    if (ptr->value == v){
      return 1;
    }
    ptr = ptr->next;
  }
  return 0;
}

struct Graph* copy_graph(struct Graph* G){
  struct Graph* G_c = (struct Graph*)malloc(sizeof(struct Graph));
  G_c->num_vertices = G->num_vertices;
  G_c->num_edges = G->num_edges;
  G_c->adj_list = (struct LinkedList*)calloc(G->num_vertices, sizeof(struct LinkedList*));
  for (int i = 0; i < G->num_vertices; i++){
    G_c->adj_list[i] = copy_linked_list(&G->adj_list[i]);
  }
  return G_c;
}

struct EdgeGenerator* edges(struct Graph* G){
  struct EdgeGenerator* EG = (struct EdgeGenerator*)malloc(sizeof(struct EdgeGenerator));
  EG->G = G;
  EG->current_u = 0;
  EG->next_v = G->adj_list[0].head;
  return EG;
}

struct Edge next_edge(struct EdgeGenerator* EG){
start:
  if (EG->current_u >= EG->G->num_vertices){
    return (struct Edge){0, 0};
  }
  if(EG->next_v == NULL){
    EG->current_u++;
    EG->next_v = EG->G->adj_list[EG->current_u].head;
    goto start;
  }
  if(EG->next_v->value < EG->current_u){
    EG->next_v = EG->next_v->next;
    goto start;
  }
  struct Edge e = {EG->current_u, EG->next_v->value};
  EG->next_v = EG->next_v->next;
  return e;
}

struct Graph* path_graph(int n){
  struct Graph* G = init_graph(n);
  for (int u = 1; u < n; u++) {
    add_edge(G, u - 1, u);
  }
  return G;
}

struct Graph* DiamondGraph(int k){
  if (k < 1){
    return NULL;
  }
  if (k == 1) {
    struct Graph* G_1 = init_graph(4);
    add_edge(G_1, 0, 1); add_edge(G_1, 1, 2);
    add_edge(G_1, 2, 3); add_edge(G_1, 3, 0);
    return G_1;
  }
  struct Graph* G_k_minus_1 = DiamondGraph(k - 1);
  struct Graph* G_k = init_graph(G_k_minus_1->num_vertices + (G_k_minus_1->num_edges * 2));
  struct EdgeGenerator* EG = edges(G_k_minus_1);
  int u = G_k_minus_1->num_vertices;
  for (struct Edge e = next_edge(EG); e.u != 0 || e.v != 0; e = next_edge(EG)){
    add_edge(G_k, e.u, u); add_edge(G_k, e.u, u + 1);
    add_edge(G_k, e.v, u); add_edge(G_k, e.v, u + 1);
    u += 2;
  }
  free_graph(G_k_minus_1);
  return G_k;
}

void print_graph(struct Graph* G) {
  for (int u = 0; u < G->num_vertices; u++){
    struct Link* ptr = G->adj_list[u].head;
    while (ptr != NULL){
      if (u < ptr->value) {
        printf("{%d, %d}", u, ptr->value);
      }
      ptr = ptr->next;
    }
    printf("\n");
  }
}
