
#ifndef CU_VECTOR_H
#define CU_VECTOR_H

typedef struct {
    int* arr;
    int cur;
    int current_size;
} Vector;

Vector* vector();

__host__ __device__
void vector_init(Vector* v);

__host__ __device__
void vector_insert(Vector* v, int elmt);

__host__
void print_vector(Vector* v);

__host__ __device__
int vector_remove(Vector* v, int elmt);

// Vector* vector_copy(Vector* v, Vector* copy);

/**
 * @return the index of elmt in v or -1 if elmt is not in v
 */ 
__host__ __device__
int vector_find(Vector* v, int elmt);

__host__ __device__
void vector_free(Vector* v);
#endif