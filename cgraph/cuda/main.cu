#include <iostream>
#include "graph.cuh"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define N 2048

static void HandleError(cudaError_t err, const char* file, int line){
  if (err != cudaSuccess){
    printf("cuda error %s at file %s in line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

void print_distance_matrix(int* hostD, int n, int limit, int start, int col_start){
  for (int i = start; i < start + limit; i++){
    // printf("counter: %d\n", host_counters[i]);
    for (int j = col_start; j < col_start + limit; j++){
      printf("%d ", hostD[i * n + j]);
      if (hostD[i * n + j] < 10){
        printf(" ");
      }
    }
    printf("\n");
  }
}
int main(){
  cudaEvent_t start, end;
  float time = 0;
  int n = N;
  struct Graph* G = path_graph(n);
  // struct Graph* devG = dev_init_graph(n);
  struct Graph* devG = copy_graph_to_device(G);
  int* D;
  HANDLE_ERROR(cudaMalloc(&D, sizeof(int) * n * n));
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  dim3 blocksize(N / 16, N / 16);
  dim3 threadsize(16, 16);
  // single_source_shortest_path<<<1, n>>>(devG, D);
  int* neighbors;
  cudaMalloc(&neighbors, sizeof(int) * n);
  get_first_neighbor<<<blocksize, threadsize>>>(devG, neighbors);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&time, start, end);
  printf("execution time of Pn apsp is %lf seconds\n", time / 1000);

  // print_device_graph(devG);
  int* hostD = (int*)malloc(n * n * sizeof(int));
  HANDLE_ERROR(cudaMemcpy(hostD, D, sizeof(int) * n * n, cudaMemcpyDeviceToHost));
  print_distance_matrix(hostD, n, 5, 250 - 2, 250 - 2);
  // int* host_neighbors = (int*)calloc(n, sizeof(int));
  // cudaMemcpy(host_neighbors, neighbors, n * sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < n; i++){
  //    printf("%d ", host_neighbors[i]);
  // }
  // print_graph(G);
  return 0;
}