#ifndef QUEUE_H
#define QUEUE_H


/**
 * A queue struct that supports enqueue, dequeue and empty predicate
 */
#ifdef VECTOR
struct Queue{
    int* arr;
    int size;
    int head;
    int tail;
};
#else
struct Queue{
    struct Link* head;
    struct Link* tail;
};
#endif


void init_queue(struct Queue* Q);


void clear(struct Queue* Q);

/**
 * insert the element e to the end of the queue
 * @param e the element to be inserted
 */
void enqueue(struct Queue* Q, int e);

/**
 * removes an element from the head of the queue ad returns it
 */
int dequeue(struct Queue* Q);

/**
 * return 1 (true) if empy, otherwise 0 (false)
 */
int is_empty(struct Queue* Q);

void free_queue(struct Queue* Q);

#endif //QUEUE_H
