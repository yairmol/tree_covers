//
// Created by Yair Molakandov on 17/07/2021.
//

#ifndef LEARNCPP_GRAPH_ALGORITHMS_H
#define LEARNCPP_GRAPH_ALGORITHMS_H

#include <iostream>
#include "Graph.h"
#include "LinkedList.h"
#include "Timer.h"

template<typename T>
struct DistanceTo {
    T to;
    int distance;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, DistanceTo<T>& d) {
  os << d.to << ": " << d.distance;
  return os;
}

template<typename T>
DistanceTo<T>* single_source_shortest_path_length_(std::set<Vertex<T>*> *adj_array, Vertex<T>* s, int num_vertices){
  Timer t;
  DistanceTo<T>* distances {new DistanceTo<T>[num_vertices]};
  bool* traversed {new bool[num_vertices]};
  distances[s->enumerator].distance = 0;
   distances[s->enumerator].to = s->value;
  queue<int>* q {init_queue(s->enumerator)};
  traversed[s->enumerator] = true;

  while (!is_empty(q)) {

    int u {dequeue(q)};
    int u_distance = distances[u].distance;

    for (Vertex<T>* v : adj_array[u]){
      if (!traversed[v->enumerator]) {
         distances[v->enumerator].to = v->value;
        distances[v->enumerator].distance = u_distance + 1;
        traversed[v->enumerator] = true;
        enqueue(q, v->enumerator);
      }
    }
  }
  double time {t.elapsed()};
  std::cout << "Elapsed time " << time << "\n";
  return distances;
}

template<typename T>
DistanceTo<T>* single_source_shortest_path_length(Graph<T>& G, T s){
  Timer t;
  std::set<Vertex<T>*>* adj_array {G.get_adj_array()};
  double time {t.elapsed()};
  std::cout << "Time building adj_array " << time << "\n";
  return single_source_shortest_path_length_(adj_array, G.get_vertex(s), G.num_vertices());
}

#endif //LEARNCPP_GRAPH_ALGORITHMS_H
