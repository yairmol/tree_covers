#ifndef CU_QUEUE_H
#define CU_QUEUE_H
/**
 * A queue struct that supports enqueue, dequeue and empty predicate
 */
typedef struct Queue{
    int* arr;
    int size;
    int head;
    int tail;
} Queue;

Queue* queue();

__host__ __device__
void init_queue(Queue* Q);

__host__ __device__
void clear(Queue* Q);

/**
 * insert the element e to the end of the queue
 * @param e the element to be inserted
 */
__host__ __device__
void enqueue(struct Queue* Q, int e);

/**
 * removes an element from the head of the queue ad returns it
 */
__host__ __device__
int dequeue(struct Queue* Q);

/**
 * return 1 (true) if empy, otherwise 0 (false)
 */
__host__ __device__
int is_empty(struct Queue* Q);

__host__ __device__
void free_queue(struct Queue* Q);

#endif //LEARNCPP_QUEUE_H
