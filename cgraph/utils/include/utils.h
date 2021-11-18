#ifndef LEARNCPP_UTILS_H
#define LEARNCPP_UTILS_H

void print_distances(int* distances, int n);

void print_sub_matrix(int** D, int row_offset, int col_offset, int limit);

/**
 * a compare function returns 1 if x > y, 0 if x == y and -1 if x < y
 */
typedef int (*comp_t)(void* x, void* y);

/** 
 * we define an iterator type as void* since any struct
 * can implement the logic of some iterator together with
 * a next function which receives an iterator and and returns the next element
 */
typedef void* iterator_t;
typedef void* (next_t)(iterator_t);

/**
 * @param iterator an iterator struct from which to choose the argmax
 * @param next a next function for the iterator struct
 * @param value_func a function that assigns a comparable value to every item in the iterator
 * @param comp a function that compares the comparable values returned by value_func
 * @return the element in iterator which maximizes value_func with respect to the comparison function comp
 */
void* argmax(iterator_t iterator, next_t next, void* (*value_func)(void*), comp_t comp);

int floatcmp(void* x, void* y);

int intcmp(void* x, void* y);

/**
 * @param iterator an iterator struct from which to choose the argmax
 * @param next a next function for the iterator struct
 * @param value_func a function that assigns a float to every item in the iterator
 * @return the element in iterator which maximizes value_func
 */
void* float_argmax(iterator_t iterator, next_t next, float (*value_func)(void*));

#endif