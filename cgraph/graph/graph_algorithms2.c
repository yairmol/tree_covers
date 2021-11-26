#define GRAPH_VEC
#include "include/graph_algorithms.h"
#include "../utils/include/queue.h"
#include "include/graph.h"
#include <stdlib.h>
#include <stdio.h>

int* single_source_shortest_path(struct IGraph* G, int s){
  int* distances = (int*)calloc(G->num_vertices, sizeof(int));
  char* visited = (char*)calloc(G->num_vertices, sizeof(char));
  distances[s] = 0;
  visited[s] = 1;
  struct Queue* Q; init_queue(Q);
  enqueue(Q, s);
  while (!is_empty(Q)){
    int u = dequeue(Q);
    int* neighbors = G->adj_list[u].arr;
    for (int i = 0; i < G->adj_list[u].cur; i++){
      int v = neighbors[i];
      if (!visited[v]){
        visited[v] = 1;
        distances[v] = distances[u] + 1;
        enqueue(Q, v);
      }
    }
  }
  free(visited);
  free_queue(Q);
  return distances;
}

int** all_pairs_shortest_path(struct IGraph* G) {
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

struct DistanceGenerator* init_distance_generator(struct IGraph* G, int start, int stop){
  struct DistanceGenerator* DG = (struct DistanceGenerator*)malloc(sizeof(struct DistanceGenerator));
  DG->G = G;
  DG->current = start;
  DG->max_node = stop;
  return DG;
}

struct DistanceGenerator* all_pairs_shortest_paths_length_generator(struct IGraph* G){
  struct DistanceGenerator* DG = (struct DistanceGenerator*)malloc(sizeof(struct DistanceGenerator));
  DG->G = G;
  DG->current = 0;
  DG->max_node = G->num_vertices;
  return DG;
}

int two_tree_embedding(struct IGraph* G_k, struct IGraph* T_1, struct IGraph* T_2, int k){
  if (k < 1) {
    return -1;
  }
  if (k == 1) {
    init_graph(G_k, 4); init_graph(T_1, 4); init_graph(T_2, 4);
    add_edge(G_k, 0, 1); add_edge(T_2, 0, 1);
    add_edge(G_k, 1, 2); add_edge(T_1, 1, 2);
    add_edge(G_k, 2, 3); add_edge(T_1, 2, 3); add_edge(T_2, 2, 3);
    add_edge(G_k, 3, 0); add_edge(T_1, 3, 0); add_edge(T_2, 3, 0);
    return 0;
  }
  struct IGraph Gkm1, Tkm1_1, Tkm1_2;
  two_tree_embedding(&Gkm1, &Tkm1_1, &Tkm1_2, k - 1);
  int k_num_vertices = Gkm1.num_vertices + (Gkm1.num_edges * 2);
  init_graph(G_k, k_num_vertices); init_graph(T_1, k_num_vertices), init_graph(T_2, k_num_vertices);
  struct EdgeGenerator* EG = edges(&Gkm1);
  int u = Gkm1.num_vertices;
  for (struct Edge e = next_edge(EG); e.u != 0 || e.v != 0; e = next_edge(EG)){
    if (e.v < e.u){
      int tmp = e.v;
      e.v = e.u;
      e.u = tmp;
    }
    add_edge(G_k, e.u, u); add_edge(G_k, e.u, u + 1);
    add_edge(G_k, e.v, u); add_edge(G_k, e.v, u + 1);
    int t_1_has_edge = has_edge(&Tkm1_1, e.u, e.v), t_2_has_edge = has_edge(&Tkm1_2, e.u, e.v);
    if (!t_1_has_edge && !t_2_has_edge){
      printf("----- %d: both t1 and t2 doesn't have an edge {%d, %d}\n", k, e.u, e.v);
    }
    if (t_1_has_edge && t_2_has_edge){
      add_edge(T_1, e.u, u + 1); add_edge(T_1, e.v, u); add_edge(T_1, e.v, u + 1);
      add_edge(T_2, e.u, u); add_edge(T_2, e.u, u + 1); add_edge(T_2, e.v, u);
    }
    else if (t_1_has_edge){
      add_edge(T_1, e.u, u + 1); add_edge(T_1, e.v, u); add_edge(T_1, e.v, u + 1);
      add_edge(T_2, e.u, u); add_edge(T_2, e.u, u + 1);
    }
    else if (t_2_has_edge){
      add_edge(T_1, e.u, u); add_edge(T_1, e.u, u + 1);
      add_edge(T_2, e.u, u); add_edge(T_2, e.v, u); add_edge(T_2, e.v, u + 1);
    }
    u += 2;
  }
  free_igraph(&Gkm1); free_igraph(&Tkm1_1); free_igraph(&Tkm1_2);
  return 0;
}