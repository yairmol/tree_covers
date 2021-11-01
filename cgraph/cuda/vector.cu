#include "vector.cuh"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Vector* vector(){
    Vector* v;
    cudaMallocManaged(&v, sizeof(Vector));
    vector_init(v);
    return v;
}

__host__ __device__
void vector_init(Vector* v){
    #ifdef __CUDA_ARCH__
    v->arr = (int*)malloc(16 * sizeof(int));
    #else
    cudaMallocManaged(&v->arr, 16 * sizeof(int));
    #endif
    v->current_size = 16;
    v->cur = 0;
}

__host__ __device__
void resize(Vector* v){
    int new_size = 2 * v->current_size;
    int* new_arr;
    #ifdef __CUDA_ARCH__
    new_arr = (int*)malloc(new_size * sizeof(int));
    #else
    cudaMallocManaged(&new_arr, new_size * sizeof(int));
    #endif
    memcpy(new_arr, v->arr, v->current_size * sizeof(int));
    #ifdef __CUDA_ARCH__
    free(v->arr);
    #else
    cudaFree(v->arr);
    #endif
    v->arr = new_arr;
    v->current_size = new_size;
}

__host__ __device__
void vector_insert(Vector* v, int elmt){
    if (v->cur == v->current_size - 1){
        resize(v);
    }
    v->arr[v->cur] = elmt;
    v->cur++;
}

__host__
void print_vector(Vector* v){
    printf("[");
    for (int i = 0; i < v->cur - 1; i++){
        printf("%d, ", v->arr[i]);
    }
    printf("%d]\n", v->arr[v->cur - 1]);
}

__host__ __device__
int vector_find(Vector* v, int elmt){
    for (int i = 0; i < v->cur; i++){
        if (v->arr[i] == elmt){
            return i;
        }
    }
    return -1;
}

__host__ __device__
int vector_remove(Vector* v, int elmt){
    int idx;
    if ((idx = vector_find(v, elmt)) != -1){
        memcpy(&v->arr[idx], &v->arr[idx + 1], sizeof(int) * (v->cur - idx - 1));
        v->cur--;
        return 1;
    }
    return 0;
}

__host__ __device__
void vector_free(Vector* v){
    #ifdef __CUDA_ARCH__
    free(v->arr);
    #else
    cudaFree(v->arr);
    #endif
}

// Vector* vector_copy(Vector* v, Vector* copy) {
//     memcpy(copy, v, sizeof(vector));
//     copy->arr = (int*)calloc(v->current_size, sizeof(int));
//     memcpy(copy->arr, v->arr, sizeof(int) * v->cur);
//     return copy;
// }