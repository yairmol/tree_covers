#include "include/queue.h"
#include "include/linked_list.h"
#include <stdlib.h>

void enqueue(struct Queue* Q, int e){
  struct Link* l = malloc(sizeof(struct Link));
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

int dequeue(struct Queue* Q){
  if (Q->head != NULL){
    int ret = Q->head->value;
    struct Link* to_free = Q->head;
    Q->head = Q->head->next;
    free(to_free);
    return ret;
  }
  return -1;
}

int is_empty(struct Queue* Q){
  return Q->head == NULL;
}

void free_queue(struct Queue* Q){
  free_link(Q->head);
  free(Q);
}