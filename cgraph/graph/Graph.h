//
// Created by Yair Molakandov on 17/07/2021.
//

#ifndef LEARNCPP_GRAPH_H
#define LEARNCPP_GRAPH_H

#include <set>
#include <map>
#include <vector>
#include <iostream>
#include "LinkedList.h"


// Edge Definition
template <typename T>
struct Edge {
    T u;
    T v;
};

template <typename T>
inline bool operator<(const Edge<T>& e1, const Edge<T>& e2){
  return e1.u < e2.u || (e1.u == e2.u && e1.v < e2.v);
}

template <typename T>
inline bool operator==(const Edge<T>& e1, const Edge<T>& e2){
  return (e1.u == e2.u && e1.v == e2.v) || (e1.u == e2.v && e1.v == e2.u);
}

// Vertex Definition
template <typename T>
struct Vertex {
    T value;
    int enumerator;
};

template <typename T>
inline bool operator<(const Vertex<T>& lhs, const Vertex<T>& rhs){
  return lhs.value < rhs.value;
}

template <typename T>
inline bool operator==(const Vertex<T>& lhs, const Vertex<T>& rhs){
  return lhs.value == rhs.value;
}


template <class T>
class Graph {
private:

    std::map<T, Vertex<T>*> vertices;
    std::map<T, std::set<Vertex<T>*>> adj_list;
    int enumeration;

public:

    Graph() : enumeration(0) {}

    std::vector<T> get_vertices(){
      std::vector<T> ret{};
      for (auto &item : vertices){
        ret.push_back(item.first);
      }
      return ret;
    }

    const std::vector<Vertex<T>*>& get_enriched_vertices() {
      std::vector<Vertex<T>*> ret{};
      for (std::pair<T, Vertex<T>*> &item : vertices){
        ret.push_back(item.second);
      }
      return ret;
    }

    const std::map<T, std::set<Vertex<T>*>>& get_adj_list(){
      return adj_list;
    }

    std::vector<Edge<T>> get_edges(){
      std::vector<Edge<T>> ret {};
      for (auto &entry : adj_list){
        for (Vertex<T> *v : entry.second){
          if (entry.first < v->value) {
            ret.push_back(Edge<T>{entry.first, v->value});
          }
        }
      }

      return ret;
    }

    Vertex<T>* add_vertex(T u) {
      if (!has_vertex(u)) {
        Vertex<T> *u_vertex{new Vertex<T>{u, enumeration}};
        enumeration++;
        vertices[u] = u_vertex;
        adj_list[u] = std::set<Vertex<T>*>{};
        return u_vertex;
      }
      return get_vertex(u);
    }

    void add_edge(T u, T v, bool add = false) {
      if (!add && !(has_vertex(u) && has_vertex(v))) {
        return;
      }
      Vertex<T>* u_vertex = add_vertex(u);
      Vertex<T>* v_vertex = add_vertex(v);
      adj_list[u].insert(v_vertex);
      adj_list[v].insert(u_vertex);
    }

    void remove_vertex(T u){
      Vertex<T>* u_vertex = get_vertex(u);
      if (has_vertex(u)){
        for (Vertex<T> &v : adj_list[u]){
          adj_list[v].erase(u_vertex);
        }
        adj_list.erase(u);
        vertices.erase(u);
      }
    }

    void remove_edge(T u, T v){
      if(has_edge(u, v)){
        adj_list[u].erase(get_vertex(v));
        adj_list[v].erase(get_vertex(u));
      }
    }

    Vertex<T>* get_vertex(T u) {
      return vertices[u];
    }

    bool has_vertex(T u){
      return vertices.find(u) != vertices.end();
    }

    bool has_edge(T u, T v){
      return adj_list[u].find(get_vertex(v)) != adj_list[u].end();
    }

    std::set<T> operator[](T u){
      return adj_list[u];
    }

    int num_vertices() {
      return vertices.size();
    }

    std::set<Vertex<T>*>* get_adj_array() {
      std::set<Vertex<T>*>* adj_array {new std::set<Vertex<T>*>[num_vertices()]};
      for (auto &entry : vertices){
        adj_array[entry.second->enumerator] = adj_list[entry.first];
      }
      return adj_array;
    }
};


template<typename T>
std::ostream& operator<<(std::ostream& os, Graph<T>& G) {
  std::cout << "V = {";
  for (auto &u : G.get_vertices()){
    std::cout << u << ", ";
  }
  std::cout << "}" << std::endl;
  std::cout << "E = {";
  std::vector<Edge<T>> edges = G.get_edges();
  for (auto &e : edges){
    std::cout << "{" << e.u << ", " << e.v << "}, ";
  }
  std::cout << "}\n";
  return os;
}

#endif //LEARNCPP_GRAPH_H
