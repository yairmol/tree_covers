#include "graph_algorithms2.h"
#include "../utils/vector_queue.h"
#include "graph_vec.h"
#include <stdlib.h>
#include <stdio.h>

int* single_source_shortest_path(struct Graph* G, int s){
  int* distances = (int*)calloc(G->num_vertices, sizeof(int));
  char* visited = (char*)calloc(G->num_vertices, sizeof(char));
  distances[s] = 0;
  visited[s] = 1;
  Queue* Q = queue();
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
  free(Q);
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