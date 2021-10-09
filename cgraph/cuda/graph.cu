#include <iostream>
#include "../cgraph/graph.h"
#include "graph.cuh"
#include "../cgraph/queue.h"
#include "queue.cuh"

struct Graph* dev_init_graph(int num_vertices){
  struct Graph* G;
  cudaMalloc(&G, sizeof(struct Graph));
  cudaMemcpy(&G->num_vertices, &num_vertices, sizeof(int), cudaMemcpyHostToDevice);
  struct LinkedList* ll;
  cudaMalloc(&ll, sizeof(struct LinkedList) * num_vertices);
  cudaMemcpy(&G->adj_list, &ll, sizeof(struct LinkedList*), cudaMemcpyHostToDevice);
  cudaMemset(ll, 0, sizeof(struct LinkedList) * num_vertices);
  return G;
}

__global__ void single_source_shortest_path(struct Graph* G, int* D){
  int s = threadIdx.x;
  int* distances = D + (s * G->num_vertices);
  char* visited = (char*)malloc(G->num_vertices * sizeof(char));
  memset(visited, 0, G->num_vertices * sizeof(char));
  distances[s] = 0;
  visited[s] = 1;
  struct Queue* Q = (struct Queue*)malloc(sizeof(struct Queue));
  Q->head = NULL;
  Q->tail = NULL;
  dev_enqueue(Q, s);
  while (!dev_is_empty(Q)){
    int u = dev_dequeue(Q);
    struct Link* ptr = G->adj_list[u].head;
    while (ptr != NULL){
      if (!visited[ptr->value]){
        visited[ptr->value] = 1;
        distances[ptr->value] = distances[u] + 1;
        dev_enqueue(Q, ptr->value);
      }
      ptr = ptr->next;
    }
  }
  free(visited);
  dev_free_queue(Q);
}

struct Graph* copy_graph_to_device(struct Graph* G){
  struct Graph* devG;
  cudaMalloc(&devG, sizeof(struct Graph));
  cudaMemcpy(&devG->num_vertices, &G->num_vertices, sizeof(int), cudaMemcpyHostToDevice);

  struct LinkedList* adj_list;
  cudaMalloc(&adj_list, sizeof(struct LinkedList) * G->num_vertices);
  cudaMemset(adj_list, 0, sizeof(struct LinkedList) * G->num_vertices);
  cudaMemcpy(&devG->adj_list, &adj_list, sizeof(struct LinkedList*), cudaMemcpyHostToDevice);

  struct LinkedList* host_adj_list = (struct LinkedList*)calloc(G->num_vertices, sizeof(struct LinkedList));

  for (int i = 0; i < G->num_vertices; i++) {
    struct Link* ptr = G->adj_list[i].head;
    struct Link* prev = NULL;
    while (ptr != NULL){
      struct Link* x;
      cudaMalloc(&x, sizeof(struct Link));
      // printf("p1: %p\t", x);
      cudaMemcpy(&x->value, &ptr->value, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(&x->next, &prev, sizeof(struct Link*), cudaMemcpyHostToDevice);
      prev = x;
      // cudaMemcpy(&x->next, devG->adj_list + i, sizeof(struct Link*), cudaMemcpyDeviceToDevice);
      // cudaMemcpy(&devG->adj_list[i].head, &x, sizeof(struct Link*), cudaMemcpyDeviceToDevice);
      ptr = ptr->next;
    }
    // printf("\n");
    host_adj_list[i].head = prev;
  }
  cudaMemcpy(adj_list, host_adj_list, sizeof(struct LinkedList) * G->num_vertices, cudaMemcpyHostToDevice);
  return devG;
}

void print_device_graph(struct Graph* devG){

  int n;
  cudaMemcpy(&n, &devG->num_vertices, sizeof(int), cudaMemcpyDeviceToHost);
  printf("n: %d\n", n);

  struct LinkedList* dev_adj_list;
  cudaMemcpy(&dev_adj_list, &devG->adj_list, sizeof(struct LinkedList*), cudaMemcpyDeviceToHost);

  struct LinkedList* adj_list = (struct LinkedList*)calloc(n, sizeof(struct LinkedList));
  cudaMemcpy(adj_list, dev_adj_list, sizeof(struct LinkedList) * n, cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i++){
    struct Link* ptr = adj_list[i].head;
    while (ptr != NULL){
      // printf("p2: %p ", ptr);
      struct Link x;
      cudaMemcpy(&x, ptr, sizeof(struct Link), cudaMemcpyDeviceToHost);
      if (i < x.value) {
        printf("{%d, %d}", i, x.value);
      }
      ptr = x.next;
    }
    printf("\n");
  }
}

__global__ void get_first_neighbor(struct Graph* G, int* neighbors){
  int u = threadIdx.x * blockDim.x + threadIdx.y;
  neighbors[u] = G->adj_list[u].head->value;
}