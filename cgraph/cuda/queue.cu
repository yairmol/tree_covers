#include "queue.cuh"

__device__ void dev_enqueue(struct Queue* Q, int e){
  struct Link* l = (struct Link*)malloc(sizeof(struct Link));
  l->value = e;
  l->next = NULL;
  if(Q->head == NULL){
    Q->head = l;
    Q->tail = l;
  } else {
    Q->tail->next = l;
    Q->tail = l;
  }
}

__device__ int dev_dequeue(struct Queue* Q){
  if (Q->head != NULL){
    int ret = Q->head->value;
    struct Link* to_free = Q->head;
    Q->head = Q->head->next;
    free(to_free);
    return ret;
  }
  return -1;
}

__device__ int dev_is_empty(struct Queue* Q){
  return Q->head == NULL;
}

__device__ void dev_free_queue(struct Queue* Q){
  dev_free_link(Q->head);
  free(Q);
}