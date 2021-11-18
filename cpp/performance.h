//
// Created by Yair Molakandov on 13/07/2021.
//
#ifndef PERFORMACE_CPP
#define PERFORMACE_CPP

#include "Timer.h"
#include <set>
#include <list>
#include <unordered_set>
#include <iostream>

// void linked_list_performance(int size){
//   Timer t;
//   LL<int>* current = nullptr;
//   for (int i{0}; i < size; i++) {
//     current = insert(current, i);
//   }
//   double time{t.elapsed()};
//   std::cout << "Creation time: " << time << std::endl;
//   t.reset();
//   LL<int>* ptr {current};
//   while (ptr != nullptr){
//     ptr = ptr->next;
//   }
//   time = t.elapsed();
//   std::cout << "Traversal time: " << time << std::endl;
// }

void set_performance(int size);

void list_performance(int size);

void unordered_set_performance(int size);

void my_set_performance(int size);

void unordered_map_performance(int size);

void my_dict_performance(int size);

#endif