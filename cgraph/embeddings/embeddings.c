#include "embeddings.h"
#include "../graph/include/graph_vec.h"
#include "../graph/include/graph_algorithms2.h"
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

float tree_embedding_distortion(struct Graph* G, struct Graph* T_1, struct Graph* T_2) {
  float max_distortion = 1, distortion;
  int min_dist;
  for (int i = 0; i < G->num_vertices; i++) {
    int *d_G = single_source_shortest_path(G, i);
    int *d_T_1 = single_source_shortest_path(T_1, i);
    int *d_T_2 = single_source_shortest_path(T_2, i);
    for (int j = i + 1; j < G->num_vertices; j++) {
      min_dist = (d_T_1[j] < d_T_2[j]) ? d_T_1[j] : d_T_2[j];
//      if (d_G[j] < min_dist) {
//        printf("vertices: (%d, %d). distance in G: %d. distance in Trees: %d\n", i, j, d_G[j], min_dist);
//      }
      distortion = ((float) min_dist) / ((float) d_G[j]);
      if (distortion > max_distortion) {
        max_distortion = distortion;
      }
    }
    free(d_G);
    free(d_T_1);
    free(d_T_2);
  }
  return max_distortion;
}

struct TEDThreadArg{
    struct DistanceGenerator* D_G;
    struct DistanceGenerator* D_T_1;
    struct DistanceGenerator* D_T_2;
    float distortion;
};

void * embedding_distortion_thread(void* arg){
  printf("thread started\n");
  struct DistanceGenerator* D_G = ((struct TEDThreadArg*)arg)->D_G;
  struct DistanceGenerator* D_T1 = ((struct TEDThreadArg*)arg)->D_T_1;
  struct DistanceGenerator* D_T2= ((struct TEDThreadArg*)arg)->D_T_2;
  int min_dist;
  int i = 0;
  float distortion, max_distortion = 1;
  for (int* d_G = next(D_G), *d_T1 = next(D_T1), *d_T2 = next(D_T2); d_G != NULL;
            d_G = next(D_G), d_T1 = next(D_T1), d_T2 = next(D_T2)) {
    for (int j = D_G->current; j < D_G->G->num_vertices; j++){
      min_dist = (d_T1[j] < d_T2[j]) ? d_T1[j] : d_T2[j];
      distortion = ((float)(min_dist)) / ((float)d_G[j]);
      if (distortion > max_distortion){
        max_distortion = distortion;
        if (distortion > 6.5){
          printf("changed max distortion to %f\n", max_distortion);
        }
      }
    }
    free(d_G); free(d_T1); free(d_T2);
    i++;
    if (i % 1000 == 0){
        printf("finished %d vertices\n", i);
    }
  }
  ((struct TEDThreadArg*)arg)->distortion = max_distortion;
  return 0;
}

float parallel_tree_embedding_distortion(struct Graph* G, struct Graph* T_1, struct Graph* T_2, int all_start, int all_stop) {
    int num_threads = 8;
    int batch_size = (all_stop - all_start) / num_threads;
    float max_distortion = 1;
    pthread_t threads[num_threads];
    struct TEDThreadArg threadArgs[num_threads];
    int start = all_start, stop = all_start + batch_size;
    for (int i = 0; /*start < G->num_vertices*/ i < num_threads; i++) {
        if (stop > G->num_vertices) {
            stop = G->num_vertices;
        }
        struct DistanceGenerator *D_G = init_distance_generator(G, start, stop);
        struct DistanceGenerator *D_T1 = init_distance_generator(T_1, start, stop);
        struct DistanceGenerator *D_T2 = init_distance_generator(T_2, start, stop);
        threadArgs[i] = (struct TEDThreadArg){D_G, D_T1, D_T2};
        printf("creating a thread from %d to %d\n", start, stop);
        pthread_create(&threads[i], NULL, embedding_distortion_thread, &threadArgs[i]);
        start = stop;
        stop += batch_size;
    }
    void *ret;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], &ret);
        printf("distortion: %f\n", threadArgs[i].distortion);
        if (threadArgs[i].distortion > max_distortion){
            max_distortion = threadArgs[i].distortion;
        }
    }
    return max_distortion;
}
