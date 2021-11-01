#include "tree_covers.cuh"
#include "graph.cuh"
#include <stdlib.h>
#include <stdio.h>

__host__
struct Graph* two_tree_embedding(struct Graph ret[3], int k){
  if (k < 1){
    return NULL;
  }
  struct Graph *G = &ret[0], *T1 = &ret[1], *T2=&ret[2];
  if (k == 1) {
    init_graph(G, 4); init_graph(T1, 4); init_graph(T2, 4);
    add_edge(G, 0, 1); add_edge(T2, 0, 1);
    add_edge(G, 1, 2); add_edge(T1, 1, 2);
    add_edge(G, 2, 3); add_edge(T1, 2, 3); add_edge(T2, 2, 3);
    add_edge(G, 3, 0); add_edge(T1, 3, 0); add_edge(T2, 3, 0);
    return ret;
  }
  struct Graph graphs[3];
  two_tree_embedding(graphs, k - 1);
  struct Graph* G_k_minus_1 = &graphs[0], *T_k_minus_1_1 = &graphs[1], *T_k_minus_1_2 = &graphs[2];
  int k_num_vertices = G_k_minus_1->num_vertices + (G_k_minus_1->num_edges * 2);
  init_graph(G, k_num_vertices); init_graph(T1, k_num_vertices); init_graph(T2, k_num_vertices);
  struct EdgeGenerator EG;
  edges(&EG, G_k_minus_1);
  int u = G_k_minus_1->num_vertices;
  for (struct Edge e = next_edge(&EG); e.u != 0 || e.v != 0; e = next_edge(&EG)){
    if (e.v < e.u){
      int tmp = e.v;
      e.v = e.u;
      e.u = tmp;
    }
    add_edge(G, e.u, u); add_edge(G, e.u, u + 1);
    add_edge(G, e.v, u); add_edge(G, e.v, u + 1);
    int t_1_has_edge = has_edge(T_k_minus_1_1, e.u, e.v), t_2_has_edge = has_edge(T_k_minus_1_2, e.u, e.v);
    if (!t_1_has_edge && !t_2_has_edge){
      printf("----- %d: both t1 and t2 doesn't have an edge {%d, %d}\n", k, e.u, e.v);
    }
    if (t_1_has_edge && t_2_has_edge){
      add_edge(T1, e.u, u + 1); add_edge(T1, e.v, u); add_edge(T1, e.v, u + 1);
      add_edge(T2, e.u, u); add_edge(T2, e.u, u + 1); add_edge(T2, e.v, u);
    }
    else if (t_1_has_edge){
      add_edge(T1, e.u, u + 1); add_edge(T1, e.v, u); add_edge(T1, e.v, u + 1);
      add_edge(T2, e.u, u); add_edge(T2, e.u, u + 1);
    }
    else if (t_2_has_edge){
      add_edge(T1, e.u, u); add_edge(T1, e.u, u + 1);
      add_edge(T2, e.u, u); add_edge(T2, e.v, u); add_edge(T2, e.v, u + 1);
    }
    u += 2;
  }
  free_graph(G_k_minus_1); free_graph(T_k_minus_1_1); free_graph(T_k_minus_1_2);
  return ret;
}