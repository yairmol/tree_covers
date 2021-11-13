#include <iostream>
// #include "GraphAlgorithms.h"
// #include "Timer.h"
#include "performance.h"
#include "hash_set.h"
#include "hash_dict.h"
#include <stdlib.h>
// void graph_construction_performance(int size) {
//   Timer t;
//   Graph<int> G{};
//   G.add_vertex(0);
//   for (int u{1}; u < size; u++) {
//     G.add_vertex(u);
//     G.add_edge(u - 1, u);
//   }
//   double time{t.elapsed()};
//   std::cout << "Graph Pn of size " << size << " construction time: " << time << "\n";
//   t.reset();
//   auto distances{single_source_shortest_path_length(G, 0)};
//   time = t.elapsed();
//   std::cout << "FastGraph Pn of size " << size << " shortest paths time: " << time << "\n";
//  std::cout << G;
//  for (int i{0}; i < size; i++) {
//    std::cout << distances[i] << "\n";
//  }
// }

int main() {
  // graph_construction_performance(10000);
//  Graph<int> G{};
//  G.add_edge(1, 2, true);
//  std::cout << G;
//  DistanceTo<int> * d = single_source_shortest_path_length(G, 1);
//  for (int i = {0}; i < 2; i++){
//    std::cout << d[i] << "\n";
//  }
  // set_performance(10000000);
  struct set s;
  std::cout << s;
  // init(s);
  for (int i = 1; i < 100; i++){
    insert(s, i);
  }
  std::cout << s;
  // std::cout << s;
  // my_set_performance(500000000);
  // unordered_set_performance(10000000);
  // void* p = malloc(1000000000);
  // std::cout << p << std::endl;
  // int size;
  // std::cin >> size;
  // std::cout << size << std::endl;
  // int* arr = new int[size];
  // for (int i = 0; i < size; i++){
  //   arr[i] = i;
  // }
  // std::cout << "[";
  // for (size_t i = 0; i < size; i++)
  // {
  //   std::cout << arr[i];
  //   if (i < size - 1){
  //     std::cout << ", ";
  //   }
  // }
  // std::cout << "]" << std::endl;

  struct dict<int, int> d;
  std::cout << d;
  for (int i = 0; i < 100; i++){
    insert(d, i, i + 1);
  }
  std::cout << d;
  return 0;
}
