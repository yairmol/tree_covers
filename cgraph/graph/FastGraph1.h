//
// Created by Yair Molakandov on 18/07/2021.
//

#ifndef LEARNCPP_FASTGRAPH1_H
#define LEARNCPP_FASTGRAPH1_H

#include "Timer.h"
#include "LinkedList.h"

template<typename T>
struct Vertex {
    T value;
    bool traversed;
};


template<typename T>
struct Graph {
    LL<Vertex<T> *> *vertices;
    LL<LL<Vertex<T> *> *> *adj_list;
};


template<typename T>
Graph<T> *empty_graph() {
  return new Graph<T>{nullptr, nullptr};
}

template<typename T>
void add_vertex(Graph<T> *G, T u) {
  G->vertices = insert(G->vertices, new Vertex<T>{u, false});
  LL<Vertex<T> *> *neighbors = nullptr;
  G->adj_list = insert(G->adj_list, neighbors);
}


template<typename T>
void add_edge(Graph<T> *G, T u, T v) {
  LL<Vertex<T> *> *vertices = G->vertices;
  LL<LL<Vertex<T> *> *> *adj_list = G->adj_list;
  Vertex<T> *u_vertex = nullptr;
  Vertex<T> *v_vertex = nullptr;
  LL<LL<Vertex<T> *> *> *u_neighbors = nullptr;
  LL<LL<Vertex<T> *> *> *v_neighbors = nullptr;
  while (vertices != nullptr) {
    if (vertices->value->value == u) {
      u_vertex = vertices->value;
      u_neighbors = adj_list;
    }
    if (vertices->value->value == v) {
      v_vertex = vertices->value;
      v_neighbors = adj_list;
    }
    vertices = vertices->next;
    adj_list = adj_list->next;
  }
  u_neighbors->value = insert(u_neighbors->value, v_vertex);
  v_neighbors->value = insert(v_neighbors->value, u_vertex);
}


template<typename T>

__attribute__((unused)) void print_graph(Graph<T> *G) {
  std::cout << "V = {";
  LL<Vertex<T> *> *vertices = G->vertices;
  while (vertices != nullptr) {
    std::cout << vertices->value->value << ", ";
    vertices = vertices->next;
  }
  std::cout << "}\nAdj = [\n";
  vertices = G->vertices;
  LL<LL<Vertex<T> *> *> *adj_list = G->adj_list;
  while (adj_list != nullptr) {
    LL<Vertex<T> *> *neighbors = adj_list->value;
    std::cout << vertices->value->value << ": [";
    while (neighbors != nullptr) {
      std::cout << neighbors->value->value << ", ";
      neighbors = neighbors->next;
    }
    std::cout << "]\n";
    adj_list = adj_list->next;
    vertices = vertices->next;
  }
  std::cout << "]\n";

}

struct FastGraph {
    Vertex<int>* * vertices;
    LL<Vertex<int>*>* * adj_list;
    int num_vertices;
};

// assume that G->vertices is 0 to n - 1 for some n
FastGraph* graph_to_fast_graph(Graph<int>* G) {
  int num_vertices = length(G->vertices);
  FastGraph* G_fast { new FastGraph{new Vertex<int>* [num_vertices], new LL<Vertex<int>*>* [num_vertices], num_vertices }};
  LL<Vertex<int> *> * vertices = G->vertices;
  LL<LL<Vertex<int> *> *> * adj_list = G->adj_list;
  while (vertices != nullptr) {
    G_fast->adj_list[vertices->value->value] = adj_list->value;
    G_fast->vertices[vertices->value->value] = vertices->value;
    vertices = vertices->next;
    adj_list = adj_list->next;
  }
  return G_fast;
}

std::ostream& operator<<(std::ostream& os, FastGraph& G) {
  os << "[\n";
  for (int u {0}; u < G.num_vertices; u++){
    LL<Vertex<int> *> *neighbors = G.adj_list[u];
    os << u << ": [";
    while (neighbors != nullptr) {
      os << neighbors->value->value << ", ";
      neighbors = neighbors->next;
    }
    os << "]\n";
  }
  os << "]\n";
  return os;
}

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

DistanceTo<int>* single_source_shortest_path_length(FastGraph *G, int s){

  DistanceTo<int>* distances {new DistanceTo<int>[G->num_vertices]};
  distances[s].distance = 0;
  distances[s].to = s;
  queue<int>* q {init_queue(s)};
  G->vertices[s]->traversed = true;

  while (!is_empty(q)) {

    int u {dequeue(q)};
    LL<Vertex<int>*>* neighbors {G->adj_list[u]};
    int u_distance = distances[u].distance;

    while (neighbors != nullptr){
      if (!neighbors->value->traversed) {
        int value = neighbors->value->value;
        distances[value].to = value;
        distances[value].distance = u_distance + 1;
        neighbors->value->traversed = true;
        enqueue(q, value);
      }
      neighbors = neighbors->next;
    }
  }
  return distances;
}


void graph_construction_performance(int size){
  Timer t;
  Graph<int> *G = empty_graph<int>();
  add_vertex(G, 0);
  for (int u{1}; u < size; u++) {
    add_vertex(G, u);
    add_edge(G, u - 1, u);
  }
  double time {t.elapsed()};
  std::cout << "Graph Pn of size " << size << " construction time: " << time << "\n";
  t.reset();
  FastGraph* G_fast = graph_to_fast_graph(G);
  time = t.elapsed();
  std::cout << "FastGraph Pn of size " << size << " construction time: " << time << "\n";
  t.reset();
  auto distances {single_source_shortest_path_length(G_fast, 0)};
  time = t.elapsed();
  std::cout << "FastGraph Pn of size " << size << " shortest paths time: " << time << "\n";
//  std::cout << *G_fast;
//  for (int i{0}; i < size; i++){
//    std::cout << distances[i] << "\n";
//  }
}


#endif //LEARNCPP_FASTGRAPH1_H
