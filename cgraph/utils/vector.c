#include "vector.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Vector* vector(){
    Vector* v = (Vector*)malloc(sizeof(Vector));
    vector_init(v);
    return v;
}

void vector_init(Vector* v){
    v->arr = (int*)calloc(16, sizeof(int));
    v->current_size = 16;
    v->cur = 0;
}

void resize(Vector* v){
    int new_size = 2 * v->current_size;
    int* new_arr = (int*)calloc(new_size, sizeof(int));
    memcpy(new_arr, v->arr, v->current_size * sizeof(int));
    free(v->arr);
    v->arr = new_arr;
    v->current_size = new_size;
}

void vector_insert(Vector* v, int elmt){
    if (v->cur == v->current_size - 1){
        resize(v);
    }
    v->arr[v->cur] = elmt;
    v->cur++;
}

void print_vector(Vector* v){
    printf("[");
    for (int i = 0; i < v->cur - 1; i++){
        printf("%d, ", v->arr[i]);
    }
    printf("%d]\n", v->arr[v->cur - 1]);
}

int vector_find(Vector* v, int elmt){
    int* arr = v->arr;
    for (int i = 0; i < v->cur; i++){
        if (v->arr[i] == elmt){
            return i;
        }
    }
    return -1;
}

int vector_remove(Vector* v, int elmt){
    int idx;
    if ((idx = vector_find(v, elmt)) != -1){
        memmove(&v->arr[idx], &v->arr[idx + 1], sizeof(int) * (v->cur - idx - 1));
        v->cur--;
        return 1;
    }
    return 0;
}

Vector* vector_copy(Vector* v, Vector* copy) {
    memcpy(copy, v, sizeof(vector));
    copy->arr = (int*)calloc(v->current_size, sizeof(int));
    memcpy(copy->arr, v->arr, sizeof(int) * v->cur);
    return copy;
}