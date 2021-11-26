#include <iostream>
#include <inttypes.h>
// #include "GraphAlgorithms.h"
// #include "Timer.h"
#include "include/performance.h"
#include "include/hash_set.h"
#include "include/hash_dict.h"
#include "include/graph.h"
#include <stdlib.h>

struct S{
  int x;
  int y;
  int z;
  // static constexpr int junk = 0;
};

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
  // struct set<int> s{0};
  // std::cout << s;
  // // // // // init(s);
  // for (int i = 1; i < 100; i++){
  //   insert(s, i);
  // }
  // std::cout << s;
  // s.set_free();
  // for (auto& i : s){
  //   std::cout << i << " ";
    // j++;
    // if (j >= 100){
    //   break;
    // }
  // }
  // std::cout << "\n";
  // std::cout << s;
  // my_set_performance(100000000);
  // unordered_set_performance(100000000);
  // my_dict_performance(100000000);
  // unordered_map_performance(100000000);
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

  struct dict<int, int> d{-1, -1};
  std::cout << d;
  for (int i = 0; i < 100; i++){
    insert(d, i, i + 1);
  }
  std::cout << d;
  d.dict_free();
  // std::cout << d[2] << "\n";
  // d[2] = 10;
  // std::cout << d[2] << "\n";

  Graph<int> G{-1};
  std::cout << G;
  // struct dict<int, set<int>> adj_dict;
  G.add_vertex(1);
  G.add_vertex(1);
  G.add_vertex(2);
  G.add_edge(1, 2, true);
  G.add_edge(1, 2, true);
  std::cout << G;
  G.free_graph();
  // std::cout << G.vertices() << "\n";
  // // int j{0};
  // for (auto& i : d){
  //   std::cout << i.k << " ";
  //   // j++;
  //   // if (j >= 100){
  //   //   break;
  //   // }
  // }
  // std::cout << "\n";
  // std::cout << sizeof(dict<int, int>) << "\n";
  // std::cout << sizeof(set<int>) << "\n";
  // std::cout << sizeof(S) << "\n";
  return 0;
}
