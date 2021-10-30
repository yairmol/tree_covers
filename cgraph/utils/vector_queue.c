#include "vector_queue.h"
#include "vector.h"
#include <string.h>

Queue* queue(){
    Queue* Q = (Queue*)malloc(sizeof(Queue));
    init_queue(Q);
    return Q;
}

void init_queue(Queue* Q){
    Q->arr = (int*)calloc(16, sizeof(int));
    Q->size = 16;
    Q->head = 0;
    Q->tail = 0;
}

void clear(Queue* Q){
    Q->head = 0;
    Q->tail = 0;
}

void resize_queue(Queue* Q){
    int new_size = 2 * Q->size;
    int* new_arr = (int*)calloc(new_size, sizeof(int));
    memcpy(new_arr, Q->arr, Q->size * sizeof(int));
    free(Q->arr);
    Q->arr = new_arr;
    Q->size = new_size;
}

/**
 * insert the element e to the end of the queue
 * @param e the element to be inserted
 */
void enqueue(struct Queue* Q, int e){
    int queue_size = Q->tail - Q->head;
    if (Q->head >= 1024){
        memmove(Q->arr, &Q->arr[Q->head], queue_size * sizeof(int));
        Q->head = 0;
        Q->tail -= 1024;
    }
    if (Q->tail >= Q->size){
        resize_queue(Q);
    }
    Q->arr[Q->tail] = e;
    Q->tail++;
}

/**
 * removes an element from the head of the queue and returns it
 */
int dequeue(struct Queue* Q){
    return Q->arr[Q->head++];
}

/**
 * return 1 (true) if empy, otherwise 0 (false)
 */
int is_empty(struct Queue* Q){
    return Q->head == Q->tail;
}

void free_queue(struct Queue* Q){
    free(Q->arr);
}
