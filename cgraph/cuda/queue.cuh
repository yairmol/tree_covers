//
// Created by yair on 08/08/2021.
//

#ifndef LEARNCPP_QUEUE_CUH
#define LEARNCPP_QUEUE_CUH

#include "linked_list.cuh"
#include "../utils/queue.h"

__device__ void dev_enqueue(struct Queue* Q, int e);

__device__ int dev_dequeue(struct Queue* Q);

__device__ int dev_is_empty(struct Queue* Q);

__device__ void dev_free_queue(struct Queue* Q);

#endif //LEARNCPP_QUEUE_CUH
