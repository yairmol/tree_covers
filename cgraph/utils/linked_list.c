#include "linked_list.h"
#include <stdlib.h>
#include <stdio.h>

typedef void (*insert_function_t)(struct LinkedList*, int);

void init_linked_list(struct LinkedList* ll){
  ll->head = NULL; ll->tail = NULL;
}

struct LinkedList* LinkedList(){
  struct LinkedList* ret = (struct LinkedList*)malloc(sizeof(struct LinkedList));
  init_linked_list(ret);
  return ret;
}

void free_link(struct Link* l){
  while (l != NULL){
    struct Link* next = l->next;
    free(l);
    l = next;
  }
}

void free_linked_list(struct LinkedList* ll){
  free_link(ll->head);
  free(ll);
}

void insert(struct LinkedList* ll, int value) {
  struct Link* l = malloc(sizeof(struct Link));
  l->value = value;
  l->next = ll->head;
  if (ll->head == NULL){
    ll->tail = l;
  }
  ll->head = l;
}

void push_back(struct LinkedList* ll, int value) {
  if (ll->head == NULL){
    return insert(ll, value);
  }
  struct Link* l = malloc(sizeof(struct Link));
  l->value = value;
  l->next = NULL;
  ll->tail->next = l;
  ll->tail = l;
}

int linked_list_remove(struct LinkedList* ll, int value){
  struct Link* prev = NULL;
  for (struct Link* cur = ll->head; cur != NULL; prev = cur, cur = cur->next){
    if (cur->value == value){
      if (prev == NULL){ // the given value is the first item in list
        ll->head = cur->next;
        if (ll->head == NULL){ // if it was the only item in themlist then clean the tail
          ll->tail = NULL;
        }
      } else {
        prev->next = cur->next;
      }
      free(cur);
      return 0;
    }
  }
  return -1;
}

struct LinkedList copy_linked_list_generic(struct LinkedList* ll, insert_function_t insert_func){
  struct LinkedList copy = {0, 0};
  struct Link* ptr = ll->head;
  while(ptr != NULL) {
    insert_func(&copy, ptr->value);
    ptr = ptr->next;
  }
  return copy;
}

struct LinkedList copy_linked_list(struct LinkedList* ll){
  return copy_linked_list_generic(ll, push_back);
}

struct LinkedList reverse_linked_list(struct LinkedList* ll){
  return copy_linked_list_generic(ll, insert);
}

struct LinkedList caten_linked_list(struct LinkedList* ll1, struct LinkedList* ll2){
  struct LinkedList ll1_copy = copy_linked_list(ll1);
  struct LinkedList ll2_copy = copy_linked_list(ll2);
  ll1_copy.tail->next = ll2_copy.head;
  return (struct LinkedList){ll1_copy.head, ll2_copy.tail};
}

void print_linked_list(struct LinkedList* ll){
  printf("linked list: [");
  for (struct Link* item = ll->head; item != NULL; item = item->next){
    printf("%d", item->value);
    if (item->next != NULL){
      printf(", ");
    }
  }
  printf("]");
}