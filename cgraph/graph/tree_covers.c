#include "tree_covers.h"
#include "graph_vec.h"
#include "graph_algorithms2.h"
#include <stdlib.h>
#include <stdio.h>

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
  free_graph(G_k_minus_1); free_graph(T_k_minus_1_1); free_graph(T_k_minus_1_2);
  free(graphs);
  struct Graph** ret = (struct Graph**)malloc(3 * sizeof(struct Graph*));
  ret[0] = G_k;
  ret[1] = T_1;
  ret[2] = T_2;
  return ret;
}

/**
 * @brief Given a separator (a subset 𝑆 ⊆ 𝑉) return a list of the induced subgraphs
 * on the connected components received by removing 𝑆
 */
struct Graph* separate(struct Graph* G, int* seperator) {
    struct Graph* G_sep = induced_subgraph(G, G.nodes.difference(seperator))
    CCs = list(nx.connected_components(G_sep))
    return [nx.Graph(nx.induced_subgraph(G, CC)) for CC in CCs]
}


struct Graph* tree_cover(struct Graph* G, find_separator_t find_separator){
    if (G->num_vertices == 1){
        return NULL;
    }
    int* separator = find_separator(G)
    subgraphs = separate(G, separator)
    all_CCs = [set(H.nodes) for H in subgraphs] + [{u} for u in separator]
    boundary_edges = get_separator_boundary_edges(G, all_CCs)
    sub_tree_covers = [tree_cover(Gi, find_separator) for Gi in subgraphs]
    bfs_trees = [nx.Graph(nx.bfs_tree(G, u)) for u in separator]
    return [
        graph_union(list(filter(None, Ts)) + [nx.Graph(boundary_edges)])
        for Ts in zip_longest(*sub_tree_covers)
    ] + bfs_trees
}