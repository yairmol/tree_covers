#include <iostream>
#include "GraphAlgorithms.h"
#include "Timer.h"

void graph_construction_performance(int size) {
  Timer t;
  Graph<int> G{};
  G.add_vertex(0);
  for (int u{1}; u < size; u++) {
    G.add_vertex(u);
    G.add_edge(u - 1, u);
  }
  double time{t.elapsed()};
  std::cout << "Graph Pn of size " << size << " construction time: " << time << "\n";
  t.reset();
  auto distances{single_source_shortest_path_length(G, 0)};
  time = t.elapsed();
  std::cout << "FastGraph Pn of size " << size << " shortest paths time: " << time << "\n";
//  std::cout << G;
//  for (int i{0}; i < size; i++) {
//    std::cout << distances[i] << "\n";
//  }
}

int main() {
  graph_construction_performance(10000);
//  Graph<int> G{};
//  G.add_edge(1, 2, true);
//  std::cout << G;
//  DistanceTo<int> * d = single_source_shortest_path_length(G, 1);
//  for (int i = {0}; i < 2; i++){
//    std::cout << d[i] << "\n";
//  }
  return 0;
}
