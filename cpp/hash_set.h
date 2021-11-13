#ifndef SET_H
#define SET_H

#include <iostream>

struct set {
    int* table;
    int size;
    int max_size;

    set(int initial_size = 16): size(0), max_size(initial_size){
        table = new int[initial_size];
        for (int i = 0; i < initial_size; i++){
            table[i] = -1;
        }
    }

    ~set(){
        delete table;
    }
};


void insert(struct set& s, int e);


bool mem(struct set& s, int e);


void remove(struct set& s, int e);


std::ostream& operator<<(std::ostream& os, struct set& s);


#endif