#include "shortest_paths.cuh"
#include "queue_vec.cuh"

__global__
void single_source_shortest_path(struct Graph* G, int offset, u_int8_t** D){
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  char* visited = (char*)malloc(G->num_vertices / 8);
  u_int8_t* Ds = D[s];
  Ds[s] = 0;
  visited[s / 8] |= 1 << (s % 8);
  Queue Q;
  init_queue(&Q);
  enqueue(&Q, s);
  while (!is_empty(&Q)){
    int u = dequeue(&Q);
    int* neighbors = G->adj_list[u].arr;
    for (int i = 0; i < G->adj_list[u].cur; i++){
      int v = neighbors[i];
      if (!visited[v]){
        visited[v / 8] |= 1 << (v % 8);
        Ds[v] = Ds[u] + 1;
        enqueue(&Q, v);
      }
    }
  }
  free(visited);
  free_queue(&Q);
}

__host__
void all_pairs_shortest_path(struct Graph* G, u_int8_t** D){
  single_source_shortest_path<<<8, 8>>>(G, 0, D);
}