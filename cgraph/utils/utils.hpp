#ifndef LEARNCPP_UTILS_H
#define LEARNCPP_UTILS_H

void print_distances(int* distances, int n);

void print_sub_matrix(int** D, int row_offset, int col_offset, int limit);

template <typename T, typename U>
using mapper_t = U (*)(T x);


template <typename T>
T min(T* elements, int size){
    T min_val = elements[0];
    for (int i = 1; i < size; i++){
        if (elements[i] < min_val){
            min_val = elements[i];
        }
    }
    return min_val;
}

template <typename T>
T max(T* elements, int size){
    T max_val = elements[0];
    for (int i = 1; i < size; i++){
        if (elements[i] > max_val){
            max_val = elements[i];
        }
    }
    return max_val;
}

template <typename T, typename U>
T argmax(T* elements, int size, mapper_t<T, U> f){
    U max_val = f(elements[0]);
    int max_idx = 0;
    for (int i = 1; i < size; i++){
        U val = f(elements[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    return max_idx;
}

template <typename T, typename U>
T argmin(T* elements, int size, mapper_t<T, U> f){
    U min_val = f(elements[0]);
    int min_idx = 0;
    for (int i = 1; i < size; i++){
        U val = f(elements[i]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }
    return min_idx;
}

/**
 * @brief maps the elements at in and stores the output at out
 * 
 * @param in an array of elements of type T
 * @param out an array of elements of type U in which the mapped results will be stored
 * @param size the number of elements in and out
 * @param f a function f:U ⟶ T
 */
template <typename T, typename U>
void map(T* in, T* out, int size, mapper_t<T, U> f) {
    for (int i = 0; i < size) {
        out[i] = f(in[i]);
    }
}

#endif