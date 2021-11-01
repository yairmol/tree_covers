#include "queue_vec.cuh"
#include "vector.cuh"
#include <string.h>

__host__
Queue* queue(){
    Queue* Q;
    cudaMallocManaged(&Q, sizeof(Queue));
    init_queue(Q);
    return Q;
}

__host__ __device__
void init_queue(Queue* Q){
    #ifdef __CUDA_ARCH__
    Q->arr = (int*)malloc(16 * sizeof(int));
    #else
    cudaMallocManaged(&Q->arr, 16 * sizeof(int));
    #endif
    Q->size = 16;
    Q->head = 0;
    Q->tail = 0;
}

__host__ __device__
void clear(Queue* Q){
    Q->head = 0;
    Q->tail = 0;
}

__host__ __device__
void resize_queue(Queue* Q){
    int new_size = 2 * Q->size;
    int* new_arr;
    #ifdef __CUDA_ARCH__
    new_arr = (int*)malloc(new_size * sizeof(int));
    #else
    cudaMallocManaged(&new_arr, sizeof(int) * new_size);
    #endif
    memcpy(new_arr, Q->arr, Q->size * sizeof(int));
    free(Q->arr);
    Q->arr = new_arr;
    Q->size = new_size;
}

/**
 * insert the element e to the end of the queue
 * @param e the element to be inserted
 */
__host__ __device__
void enqueue(struct Queue* Q, int e){
    int queue_size = Q->tail - Q->head;
    if (Q->head >= 1024){
        memcpy(Q->arr, &Q->arr[Q->head], queue_size * sizeof(int));
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
__host__ __device__
int dequeue(struct Queue* Q){
    return Q->arr[Q->head++];
}

/**
 * return 1 (true) if empy, otherwise 0 (false)
 */
__host__ __device__
int is_empty(struct Queue* Q){
    return Q->head == Q->tail;
}

__host__ __device__
void free_queue(struct Queue* Q){
    #ifdef __CUDA_ARCH__
    free(Q->arr);
    #else
    cudaFree(Q->arr);
    #endif
}
