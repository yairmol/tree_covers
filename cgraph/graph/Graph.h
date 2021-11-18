//
// Created by Yair Molakandov on 17/07/2021.
//

#ifndef LEARNCPP_GRAPH_H
#define LEARNCPP_GRAPH_H

#include <iostream>
#include "../../cpp/hash_dict.h"
#include "../../cpp/hash_set.h"


// Edge Definition
template <typename T>
struct edge {
    T u;
    T v;
};

template <typename T>
inline bool operator<(const edge<T>& e1, const edge<T>& e2){
  return e1.u < e2.u || (e1.u == e2.u && e1.v < e2.v);
}

template <typename T>
inline bool operator==(const edge<T>& e1, const edge<T>& e2){
  return (e1.u == e2.u && e1.v == e2.v) || (e1.u == e2.v && e1.v == e2.u);
}


template <typename T>
inline int operator%(const edge<T>& e, int x){
  return (e.u % x + e.v % x) % x;
}


template <typename T>
class Graph {
private:
  struct set<T> V;
  struct dict<T, struct set<T>> adj_dict;
  struct set<edge<T>> edges;
public:

    Graph() {}

    const struct set& vertices(){
      return V;
    }
    // struct set<edge> get_edges(){
    //   struct set<edge> edges;
    //   for (auto& kv : adj_dict){
    //     for (auto& v: kv.v) {
    //       insert(edges, edge<T>{kv.k, v});
    //     }
    //   }
    //   return edges;
    // }

    struct set<edge<T>>& get_edges(){
      return edges;
    }

    void add_vertex(T u) {
      insert(vertices, u);
    }

    void add_edge(T u, T v, bool add = false) {
      if (!add && !(has_vertex(u) && has_vertex(v))) {
        return;
      }
      add_vertex(u);
      add_vertex(v);
      insert(adj_list[u], v);
      insert(adj_list[v], u);
      insert(edges, edge<T>{u, v})
    }

    void remove_vertex(T u){
      return;
    }

    void remove_edge(T u, T v){
      return;
    }

    bool has_vertex(T u){
      return mem(vertices u);
    }

    bool has_edge(T u, T v){
      return mem(adj_list[u], v);
    }

    struct set<T>& operator[](T u){
      return adj_list[u];
    }

    int num_vertices() {
      return vertices.size;
    }

    int num_edges(){
      return edges.size;
    }
};


template<typename T>
std::ostream& operator<<(std::ostream& os, Graph<T>& G) {
  os << "V = {";
  for (auto &u : G.get_vertices()){
    os << u << ", ";
  }
  os << "}" << std::endl;
  os << "E = {";
  std::vector<Edge<T>> edges = G.get_edges();
  for (auto &e : edges){
    os << "{" << e.u << ", " << e.v << "}, ";
  }
  os << "}\n";
  return os;
}

#endif //LEARNCPP_GRAPH_H
