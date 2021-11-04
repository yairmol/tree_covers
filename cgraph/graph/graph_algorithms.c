#include "graph_algorithms.h"
#include "../utils/linked_list.h"
#include "../utils/queue.h"
#include "graph.h"
#include <stdlib.h>
#include <stdio.h>

int* single_source_shortest_path(struct Graph* G, int s){
  int* distances = (int*)calloc(G->num_vertices, sizeof(int));
  char* visited = (char*)calloc(G->num_vertices, sizeof(char));
  distances[s] = 0;
  visited[s] = 1;
  struct Queue* Q = (Queue*)calloc(1, sizeof(struct Queue));
  enqueue(Q, s);
  while (!is_empty(Q)){
    int u = dequeue(Q);
    struct Link* ptr = G->adj_list[u].head;
    while (ptr != NULL){
      if (!visited[ptr->value]){
        visited[ptr->value] = 1;
        distances[ptr->value] = distances[u] + 1;
        enqueue(Q, ptr->value);
      }
      ptr = ptr->next;
    }
  }
  free(visited);
  free_queue(Q);
  return distances;
}

int** all_pairs_shortest_path(struct Graph* G) {
  int **D = (int**)calloc(G->num_vertices, sizeof(int *));
  for (int i = 0; i < G->num_vertices; i++) {
    D[i] = single_source_shortest_path(G, i);
  }
  return D;
}

int* next(struct DistanceGenerator* DG){
  if (DG->current >= DG->max_node){
    return NULL;
  }
  int* ret = single_source_shortest_path(DG->G, DG->current);
  DG->current++;
  return ret;
}

struct DistanceGenerator* init_distance_generator(struct Graph* G, int start, int stop){
  struct DistanceGenerator* DG = (struct DistanceGenerator*)malloc(sizeof(struct DistanceGenerator));
  DG->G = G;
  DG->current = start;
  DG->max_node = stop;
  return DG;
}

struct DistanceGenerator* all_pairs_shortest_paths_length_generator(struct Graph* G){
  struct DistanceGenerator* DG = (struct DistanceGenerator*)malloc(sizeof(struct DistanceGenerator));
  DG->G = G;
  DG->current = 0;
  DG->max_node = G->num_vertices;
  return DG;
}

struct Graph** two_tree_embedding(int k){
  if (k < 1){
    return NULL;
  }
  if (k == 1) {
    struct Graph* G_1 = init_graph(4), *T_1 = init_graph(4), *T_2 = init_graph(4);
    add_edge(G_1, 0, 1); add_edge(T_2, 0, 1);
    add_edge(G_1, 1, 2); add_edge(T_1, 1, 2);
    add_edge(G_1, 2, 3); add_edge(T_1, 2, 3); add_edge(T_2, 2, 3);
    add_edge(G_1, 3, 0); add_edge(T_1, 3, 0); add_edge(T_2, 3, 0);
    struct Graph** ret = (struct Graph**)malloc(3 * sizeof(struct Graph*));
    ret[0] = G_1;
    ret[1] = T_1;
    ret[2] = T_2;
    return ret;
  }
  struct Graph** graphs = two_tree_embedding(k - 1);
  struct Graph* G_k_minus_1 = graphs[0], *T_k_minus_1_1 = graphs[1], *T_k_minus_1_2 = graphs[2];
  int k_num_vertices = G_k_minus_1->num_vertices + (G_k_minus_1->num_edges * 2);
  struct Graph* G_k = init_graph(k_num_vertices), *T_1 = init_graph(k_num_vertices), *T_2 = init_graph(k_num_vertices);
  struct EdgeGenerator* EG = edges(G_k_minus_1);
  int u = G_k_minus_1->num_vertices;
  for (struct Edge e = next_edge(EG); e.u != 0 || e.v != 0; e = next_edge(EG)){
    if (e.v < e.u){
      int tmp = e.v;
      e.v = e.u;
      e.u = tmp;
    }
    add_edge(G_k, e.u, u); add_edge(G_k, e.u, u + 1);
    add_edge(G_k, e.v, u); add_edge(G_k, e.v, u + 1);
    int t_1_has_edge = has_edge(T_k_minus_1_1, e.u, e.v), t_2_has_edge = has_edge(T_k_minus_1_2, e.u, e.v);
    if (t_1_has_edge && t_2_has_edge){
      add_edge(T_1, e.u, u); add_edge(T_1, e.u, u + 1); add_edge(T_1, e.v, u);
      add_edge(T_2, e.v, u); add_edge(T_2, e.v, u + 1); add_edge(T_2, e.u, u);
    }
    else if (t_1_has_edge){
      add_edge(T_1, e.u, u); add_edge(T_1, e.u, u + 1); add_edge(T_1, e.v, u);
      add_edge(T_2, e.u, u); add_edge(T_2, e.u, u + 1);
    }
    else if (t_2_has_edge){
      add_edge(T_1, e.u, u); add_edge(T_1, e.u, u + 1);
      add_edge(T_2, e.u, u); add_edge(T_2, e.u, u + 1); add_edge(T_2, e.v, u);
    }
    u += 2;
  }
  free_graph(G_k_minus_1); free_graph(T_k_minus_1_1); free_graph(T_k_minus_1_2);
  free(graphs);
  struct Graph** ret = (struct Graph**)malloc(3 * sizeof(struct Graph*));
  ret[0] = G_k;
  ret[1] = T_1;
  ret[2] = T_2;
  return ret;
}