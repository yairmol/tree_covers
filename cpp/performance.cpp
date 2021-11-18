//
// Created by Yair Molakandov on 13/07/2021.
//
#include "performance.h"
#include "Timer.h"
#include <set>
#include "hash_set.h"
#include "hash_dict.h"
#include <list>
#include <unordered_set>
#include <unordered_map>
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

void set_performance(int size){
  Timer t;
  std::set<int> s;
  for (int i{0}; i < size; i++){
    s.insert(i);
  }
  double time = t.elapsed();
  std::cout << "Set build in " << time << " seconds. set size is " << s.size() << std::endl;
  t.reset();
  // for (int _ : s);
  time = t.elapsed();
  std::cout << "Set iterated in " << time << " seconds" << std::endl;
}


void my_set_performance(int size){
  Timer t;
  struct set<int> s{-1};
  // init(s);
  for (int i{0}; i < size; i++){
    insert(s, i);
  }
  double time = t.elapsed();
  std::cout << "Set build in " << time << " seconds. set size is " << s.size << std::endl;
  t.reset();
  bool ismem = true;
  for (int i = 0; i < size; i++){
    ismem = ismem && mem(s, i);
  }
  std::cout << ismem << std::endl;
  time = t.elapsed();
  std::cout << "Set iterated in " << time << " seconds" << std::endl;
  s.set_free();
  // t.reset();
  // ismem = false;
  // for (int i = size; i < 1000; i++){
  //   ismem = ismem || mem(s, i);
  // }
  // std::cout << ismem << std::endl;
  // time = t.elapsed();
  // std::cout << "1000 items not in set run in " << time << " seconds" << std::endl;
}


void list_performance(int size) {
  Timer t;
  std::list<int> l;
  for (int i{0}; i < size; i++) {
    l.push_back(i);
  }
  double time = t.elapsed();
  std::cout << "List build in " << time << " seconds. set size is " << l.size() << std::endl;
  t.reset();
  // for (int i : l);
  time = t.elapsed();
  std::cout << "List iterated in " << time << " seconds" << std::endl;
}

void unordered_set_performance(int size){
  Timer t;
  std::unordered_set<int> s;
  for (int i{0}; i < size; i++){
    s.insert(i);
  }
  double time = t.elapsed();
  std::cout << "Set build in " << time << " seconds. set size is " << s.size() << std::endl;
  double total_time {0.0};
  t.reset();
  for (int i{0}; i < size; i++){
    s.find(i);
  }
  total_time = t.elapsed();
  std::cout << "Set iterated in " << total_time << " seconds" << std::endl;
}


void unordered_map_performance(int size){
  Timer t;
  std::unordered_map<int, int> d;
  for (int i{0}; i < size; i++){
    d.insert(std::pair<int, int>{i, i + 1});
  }
  double time = t.elapsed();
  std::cout << "Dict build in " << time << " seconds. set size is " << d.size() << std::endl;
  double total_time {0.0};
  t.reset();
  for (int i{0}; i < size; i++){
    d.find(i);
  }
  total_time = t.elapsed();
  std::cout << "Dict iterated in " << total_time << " seconds" << std::endl; 
}

void my_dict_performance(int size){
  Timer t;
  struct dict<int, int> d{-1, -1};
  for (int i{0}; i < size; i++){
    insert(d, i, i + 1);
  }
  double time = t.elapsed();
  std::cout << "Dict build in " << time << " seconds. set size is " << d.size << std::endl;
  double total_time {0.0};
  t.reset();
  for (int i{0}; i < size; i++){
    mem(d, i);
  }
  total_time = t.elapsed();
  std::cout << "Dict iterated in " << total_time << " seconds" << std::endl; 
}