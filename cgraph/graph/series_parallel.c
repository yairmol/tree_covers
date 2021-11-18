#include "include/graph.h"
#include "../utils/include/queue.h"
#include "../utils/include/linked_list.h"
#include <stdlib.h>
#include <stdio.h>
#include "include/series_parallel.h"
#include "../embeddings/embeddings.h"
#include "../utils/include/utils.h"


typedef int (*shift_t)(int, SPGraph*, SPGraph*);


SPGraph* init_sp_graph(int num_vertices, int s, int t){
  SPGraph* ret = (SPGraph*)malloc(sizeof(SPGraph));
  ret->graph = init_graph(num_vertices);
  ret->s = s; ret->t = t;
  return ret;
}


SPGraph* K2_sp_graph(){
  SPGraph* ret = init_sp_graph(2, 0, 1);
  add_edge(ret->graph, 0, 1);
  return ret;
}


// shifts a vertex to its new representing integer
// with respect to parallel composition.
// assume that G2->s < G2->t
int pshift(int u, SPGraph* G1, SPGraph* G2){
  if (u < G2->s) return u + G1->graph->num_vertices;
  if (u == G2->s) return G1->s;
  if (u < G2->t) return u + G1->graph->num_vertices - 1;
  if (u == G2->t) return G1->t;
  return u + G1->graph->num_vertices - 2;
}

// shifts a vertex to its new representing integer
// with respect to series composition
int sshift(int u, SPGraph* G1, SPGraph* G2){
  if (u < G2->s) return u + G1->graph->num_vertices;
  if (u == G2->s) return G1->t;
  return u + G1->graph->num_vertices - 1;
}


SPGraph* generic_composition(SPGraph* G1, SPGraph* G2, SPGraph* G, shift_t shift){
  struct EdgeGenerator* E1 = edges(G1->graph);
  for (Edge e = next_edge(E1); e.u != 0 || e.v != 0; e = next_edge(E1)){
      add_edge(G->graph, e.u, e.v);
  }

  struct EdgeGenerator* E2 = edges(G2->graph);
  for (Edge e = next_edge(E2); e.u != 0 || e.v != 0; e = next_edge(E2)){
      add_edge(G->graph, shift(e.u, G1, G2), shift(e.v, G1, G2));
  }
  return G;
}

// parallel composition of series-parallel graphs
SPGraph* parallel_composition(SPGraph* G1, SPGraph* G2){
  SPGraph* G = init_sp_graph(G1->graph->num_vertices +
                              G2->graph->num_vertices - 2,
                              G1->s,
                              G1->t);

  return generic_composition(G1, G2, G, pshift);
}

// series compositio of series parallel graphs
SPGraph* series_composition(SPGraph* G1, SPGraph* G2){
  int num_vertices = G1->graph->num_vertices + G2->graph->num_vertices - 1;
  SPGraph* G = init_sp_graph(num_vertices, G1->s, sshift(G2->t, G1, G2));
  return generic_composition(G1, G2, G, sshift);
}


DecompTree* diamond_graph_decomp_tree(int k){
  if (k < 0) return NULL;
  if (k == 0){
    DecompTree* ret = (DecompTree*)malloc(sizeof(DecompTree));
    ret->composition = K2; ret->left = NULL; ret->right = NULL;
    return ret;
  }
  DecompTree* series_comp = (DecompTree*)malloc(sizeof(DecompTree));
  series_comp->composition = SERIES_COMPOSITION;
  series_comp->left = series_comp->right = diamond_graph_decomp_tree(k - 1);
  DecompTree* ret = (DecompTree*)malloc(sizeof(DecompTree));
  ret->composition = PARALLEL_COMPOSITION;
  ret->left = ret->right = series_comp;
  return ret;
}


char* composition_to_string(enum Composition comp){
  switch (comp){
    case K2:
        return "K2";
        break;
    case SERIES_COMPOSITION:
        return "S";
        break;
    case PARALLEL_COMPOSITION:
        return "P";
        break;
    default:
        break;
  }
}

int max(int x, int y){
  return x < y ? y : x;
}


int height(DecompTree* T){
  if (T == NULL) return 0;
  return max(height(T->left), height(T->right)) + 1;
}


// void print_decomp_tree(DecompTree* T){
//     int tree_height = height(T);
//     int indent_len = 0;
//     struct Queue to_print;
//     enqueue(&to_print, T->composition);
//     while (!is_empty(&to_print)){
//         DecompTree* cur = dequeue(&to_print);
//         if (cur->left != NULL) enqueue(&to_print, cur->left);
//         if (cur->right != NULL) enqueue(&to_print, cur->right);
//         printf("%s\n", composition_to_string(cur->composition));
//     }
// }

typedef struct LinkedList path_t;

typedef struct SPGraphWP {
    SPGraph G;
    path_t st_path;
} SPGraphWP;


// shifts a path in place
void shift_path(path_t* path, shift_t shift, SPGraph* G1, SPGraph* G2){
  for (struct Link* u = path->head; u != NULL; u = u->next){
    u->value = shift(u->value, G1, G2);
  }
}

path_t join_paths(path_t* p1, path_t* p2){
  if (p2->tail->value == p1->head->value){
    path_t* tmp = p1;
    p1 = p2;
    p2 = tmp;
  }
  if (p1->tail->value != p2->head->value){
    fprintf(stderr, "Errot at %s:%d at function join paths. paths are not overlapping", __FILE__, __LINE__);
    return (path_t){NULL, NULL};
  }
  p2->head = p2->head->next;
  return caten_linked_list(p1, p2);
}

struct Edge path_edge(struct Link* u){
  return (struct Edge){u->value, u->next->value};
}

typedef struct EmbDistArg{
  Graph* G;
  Graph* T1;
  Graph* T2;
} EmbDistArg;

void* embedding_distortion_wrapper(void* arg){
  EmbDistArg* info = (EmbDistArg*)arg;
  float* distortion = (float*)malloc(sizeof(float));
  *distortion = tree_embedding_distortion(info->G, info->T1, info->T2);
  return (void*)distortion;
}

typedef struct EdgeRemoveIterator {
  Graph* G;
  Edge* edges;
  int num_edges;
  int current;
} EdgeRemoveIterator;

EdgeRemoveIterator* edge_remove_iterator_from_path(Graph* G, path_t* path){
  EdgeRemoveIterator* eri = (EdgeRemoveIterator*)malloc(sizeof(EdgeRemoveIterator));
  eri->G = G;
  eri->current = 0;
  struct Link* ptr = path->head;
  int pathlen = 0;
  for (struct Link* ptr = path->head; ptr != NULL; ptr = ptr->next) { pathlen++; }
  eri->edges = (Edge*)calloc(pathlen - 1, sizeof(Edge));
  eri->num_edges = pathlen - 1;
  int i = 0;
  for (Edge e = path_edge(ptr); e.u != 0 || e.v != 0; ptr = ptr->next){
    eri->edges[i] = e;
  }
}

// args:
//   1. EdgeRemoveGenerator* iter
// returns: Graph*
void* edge_remove_iterator_next(void* iter){
  EdgeRemoveIterator* eri = (EdgeRemoveIterator*)iter;
  if (eri->current > 0){
    Edge prev = eri->edges[eri->current - 1];
    add_edge(eri->G, prev.u, prev.v);
  }
  if (eri->current >= eri->num_edges){
    return NULL;
  }
  Edge curr = eri->edges[eri->current];
  remove_edge(eri->G, curr.u, curr.v);
  eri->current++;
  return eri->G;
}

void sp_tree_cover_embedding(DecompTree* T, SPGraphWP* T1, SPGraphWP* T2){
  if (T == NULL){
    *T1 = (SPGraphWP){{NULL, 0, 0}, {NULL}};
    *T2 = *T1;
    return;
  }
  if (T->composition == K2){
    path_t k2; 
    init_linked_list(&k2);
    push_back(&k2, 0); push_back(&k2, 1);
    *T1 = (SPGraphWP){*K2_sp_graph(), k2};
    *T2 = *T1;
    return;
  }
  SPGraphWP T1_left, T2_left, T1_right, T2_right;
  sp_tree_cover_embedding(T->left, &T1_left, &T2_left);
  sp_tree_cover_embedding(T->right, &T1_right, &T2_right);
  if (T->composition == SERIES_COMPOSITION){
    T1_right.st_path.head = T1_right.st_path.head->next;
    shift_path(&T1_right.st_path, sshift, &T1_left.G, &T1_right.G);
    *T1 = (SPGraphWP){
        *series_composition(&T1_left.G, &T1_right.G),
        caten_linked_list(&T1_left.st_path, &T1_right.st_path)
    };
    *T2 = (SPGraphWP){
        *series_composition(&T2_left.G, &T2_right.G),
        caten_linked_list(&T2_left.st_path, &T2_right.st_path)
    };
    return;
  }
  // T1->G = parallel_composition(&T1_left.G, &T1_right.G);
  // T2->G = parallel_composition(&T2_left.G, &T2_right.G);
  // path_t cycle = 
  // for (struct Link* u = )
  // TODO: write the case of parallel composition
}