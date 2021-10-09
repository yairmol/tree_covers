//
// Created by Yair Molakandov on 13/07/2021.
//
#include <set>
#include <list>

void linked_list_performance(int size){
  Timer t;
  LL<int>* current = nullptr;
  for (int i{0}; i < size; i++) {
    current = insert(current, i);
  }
  double time{t.elapsed()};
  std::cout << "Creation time: " << time << std::endl;
  t.reset();
  LL<int>* ptr {current};
  while (ptr != nullptr){
    ptr = ptr->next;
  }
  time = t.elapsed();
  std::cout << "Traversal time: " << time << std::endl;
}

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
  for (int i{0}; i < size; i++){
    t.reset();
    s.find(i);
    total_time += t.elapsed();
  }
  std::cout << "Set iterated in " << total_time << " seconds" << std::endl;
}