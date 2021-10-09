#include "linked_list.cuh"

__device__ void dev_free_link(struct Link* l){
  while (l != NULL){
    struct Link* next = l->next;
    free(l);
    l = next;
  }
}

__device__ void dev_insert(struct LinkedList* ll, int value) {
  struct Link* l = (struct Link*)malloc(sizeof(struct Link));
  l->value = value;
  l->next = ll->head;
  ll->head = l;
}