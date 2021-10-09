
#ifndef LEARNCPP_VECTOR_H
#define LEARNCPP_VECTOR_H

#include <stdlib.h>

typedef struct {
    int* arr;
    int cur;
    int current_size;
} Vector;

Vector* vector();

void vector_init(Vector* v);

void vector_insert(Vector* v, int elmt);

void print_vector(Vector* v);

int vector_remove(Vector* v, int elmt);

Vector* vector_copy(Vector* v, Vector* copy);

/**
 * @return the index of elmt in v or -1 if elmt is not in v
 */ 
int vector_find(Vector* v, int elmt);


#endif