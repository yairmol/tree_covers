//
// Created by yair on 08/08/2021.
//

#ifndef LEARNCPP_LINKED_LIST_CUH
#define LEARNCPP_LINKED_LIST_CUH
#include "../cgraph/linked_list.h"

__device__ void dev_free_link(struct Link* l);

__device__ void dev_insert(struct LinkedList* ll, int value);
#endif //LEARNCPP_LINKED_LIST_CUH
