#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>

void print_distances(int* distances, int n){
  for (int i = 0; i < n; i++){
    if (i % 5 == 0){
      printf("\n");
    }
    printf("d(%d, %d) = %d\t", 0, i, distances[i]);
  }
  printf("\n");
}

void print_sub_matrix(int** D, int row_offset, int col_offset, int limit){
  for (int i = row_offset; i < row_offset + limit; i++) {
    for (int j = col_offset; j < col_offset + limit; j++) {
      printf("%d ", D[i][j]);
    }
    printf("\n");
  }
}

//typedef struct {
//    int (*f)(int);
//    int n;
//    int m;
//} IntEmbedding;
//
//typedef int (*Metric)(int, int);

/*float embedding_distortion(IntEmbedding Emb, Metric d1, Metric d2){
  float max_distortion = 1, distortion;
  for (int i = 0; i < Emb.n; i++){
    int f_i = Emb.f(i);
    for (int j = i + 1; j < Emb.m; j++){
      distortion = (float)d2(f_i, Emb.f(j)) / (float)d1(i, j);
      if (distortion > max_distortion){
        max_distortion = distortion;
      }
    }
  }
  return max_distortion;
}*/

// void all_pairs_shortest_paths_performance(){
//   clock_t t, midt;
//   t = clock();
//   struct Graph* G = DiamondGraph(7);
//   t = clock() - t;
//   double build_time = ((double)t)/CLOCKS_PER_SEC;
//   printf("time taken to build P%d: %f\n", G->num_vertices, build_time);
//   t = clock();
//   midt = t;
//   struct DistanceGenerator* DG = all_pairs_shortest_paths_length_generator(G);
//   int i = 0;
//   for (int* distances = next(DG); distances != NULL; distances = next(DG)){
//     free(distances);
//     if (i % 1000 == 0){
//       printf("reached %d in %f\n", i, ((double)(clock() - midt))/CLOCKS_PER_SEC);
//       midt = clock();
//     }
//     i++;
//   }
//   t = clock() - t;
//   double apsp_time = ((double)t)/CLOCKS_PER_SEC;
//   printf("time taken to run apsp P%d: %f\n", G->num_vertices, apsp_time);
//   free_graph(G);
// }

// struct ThreadArg{
//     struct DistanceGenerator* DG;
// };

// void * distance_thread(void* arg){
//   struct DistanceGenerator* DG = ((struct ThreadArg*)arg)->DG;
//   for (int* distances = next(DG); distances != NULL; distances = next(DG)) {
//     free(distances);
//   }
//   return 0;
// }

//void parallel_all_pairs_shortest_paths_performance(){
//  struct timespec start, finish;
//  clock_gettime(CLOCK_MONOTONIC, &start);
//  struct Graph* G = DiamondGraph(9);
//  clock_gettime(CLOCK_MONOTONIC, &finish);
//  double elapsed = (double)(finish.tv_sec - start.tv_sec);
//  elapsed += (double)(finish.tv_nsec - start.tv_nsec) / 1000000000.0;
//  printf("time taken to build P%d: %f\n", G->num_vertices, elapsed);
//  int num_threads = 8 * 128;
//  int batch_size = G->num_vertices / num_threads;
//  clock_gettime(CLOCK_MONOTONIC, &start);
//  pthread_t tids[num_threads];
//  int v_start = 0, stop = batch_size;
//  for (int i = 0; i < num_threads; i++) {
//    if (stop > G->num_vertices) {
//      stop = G->num_vertices;
//    }
//    struct DistanceGenerator *DG = init_distance_generator(G, v_start, stop);
//    struct ThreadArg* ta = (struct ThreadArg*)malloc(sizeof(struct ThreadArg));
//    *ta = (struct ThreadArg){DG};
//    pthread_create(&tids[i], NULL, distance_thread, ta);
//    v_start = stop;
//    stop += batch_size;
//  }
//  void *ret;
//  for (int i = 0; i < num_threads; i++) {
//    pthread_join(tids[i], &ret);
//  }
//  clock_gettime(CLOCK_MONOTONIC, &finish);
//  elapsed = (double)(finish.tv_sec - start.tv_sec);
//  elapsed += (double)(finish.tv_nsec - start.tv_nsec) / 1000000000.0;
//  printf("time taken to run apsp P%d: %f\n", G->num_vertices, elapsed);
//  free_graph(G);
//}
int floatcmp(void* x, void* y){
  float xf = *((float*)x);
  float yf = *((float*)y);
  return x < y ? -1 : x > y ? 1 : 0;
}

int intcmp(void* x, void* y){
  int xf = *((int*)x);
  int yf = *((int*)y);
  return x < y ? -1 : x > y ? 1 : 0;
}

// return 0 if equal,
// 1 if the first is bigger
// -1 if the first is smaller

// void* argmax(iterator_t iterator, next_t next, void* (*value_func)(void*), comp_t comp){
//   void* max_element = next(iterator);
//   void* max_value = value_func(iterator);
//   for (void* elmt = next(iterator); elmt != NULL; elmt = next(iterator)){
//     void* value = value_func(elmt);
//     if (comp(value, max_value) > 0){
//       max_value = value;
//       max_element = elmt;
//     }
//   }
//   return max_element;
// }

// void* float_argmax(iterator_t iterator, next_t next, float (*value_func)(void*)){
//   void* max_element = next(iterator);
//   float max_value = value_func(iterator);
//   for (void* elmt = next(iterator); elmt != NULL; elmt = next(iterator)){
//     float value = value_func(elmt);
//     if (value > max_value){
//       max_value = value;
//       max_element = elmt;
//     }
//   }
//   return max_element;
// }

template <typename T, typename U>
void map(T* in, T* out, int size, mapper_t<T, U> f) {
    for (int i = 0; i < size) {
        out[i] = f(in[i]);
    }
}