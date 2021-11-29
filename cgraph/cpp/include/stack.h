#ifndef CPP_STACK_H
#define CPP_STACK_H

#include <stdlib.h>
#include <iostream>

template <typename T>
struct stack {
private:
    T* arr;
    int max_size;

    void resize(){
        T* old_arr = arr;
        max_size *= 2;
        arr = (T*)calloc(max_size, sizeof(T));
        for (int i = 0; i < size; i++){
            arr[i] = old_arr[i];
        }
        free(old_arr);
    }

public:
    int size;

    stack(): max_size(2), size(0) {
        arr = (T*)calloc(max_size, sizeof(T));
    }

    void push(T e){
        if (size >= max_size){
            resize();
        }
        arr[size++] = e;
    }

    T pop(){
        if (size > 0) {
            return arr[--size];
        }
        return arr[0]; // TODO: chage
    }

    T peek(int i = 0){
        return arr[size - 1 - i];
    }

    void print(std::ostream& os){
        for (int i = size - 1; i >= 0; i--) {
            os << arr[i] << " ";
        }
        os << "\n";
    }

    bool is_empty(){
        return size == 0;
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, stack<T>& s){
    s.print(os);
    return os;
}

#endif // CPP_STACK_H