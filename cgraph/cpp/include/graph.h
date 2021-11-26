#ifndef CPP_GRAPH_H
#define CPP_GRAPH_H

#include <iostream>
#include "hash_dict.h"
#include "hash_set.h"
extern "C"{
  #include "../../graph/include/graph.h"
}

// Edge Definition
template <typename T>
struct edge {
  T u;
  T v;

  edge(){}
  edge(T u, T v): u(u), v(v) {}
  edge(int def): u(0), v(0){}

  bool operator!=(const edge<T>& e){
    return u != e.u || v != e.v;
  }

  bool operator==(const edge<T>& e){
    return u == e.u && v == e.v || u == e.v && v == e.u;
  }

  int operator%(int x){
    return (u % x + v % x) % x;
  }

  bool operator<(const edge<T>& other){
    return u < other.u || (u == other.u && v < other.v);
  }
};


template <typename T>
class Graph {
private:
  struct set<T> V;
  struct dict<T, struct set<T>*> adj_dict;
  struct set<edge<T>> edges;
  struct dict<int, T> imapping;
  struct dict<T, int> reverse_mapping;
  T junkval;
public:
    Graph(T junkval) : 
      junkval(junkval),
      V{junkval},
      adj_dict{-1, nullptr},
      edges{edge<T>{junkval, junkval}},
      imapping{-1, junkval},
      reverse_mapping{junkval, -1} {
    }

    void free_graph(){
      V.set_free();
      edges.set_free();
      for (kvpair<T, struct set<T>*> kv : adj_dict){
        kv.v->set_free();
        free(kv.v);
      }
      adj_dict.dict_free();
    }

    struct set<T>& vertices(){
      return V;
    }

    struct set<edge<T>>& get_edges(){
      return edges;
    }

    void add_vertex(T u) {
      if (!mem(V, u)){
        insert(V, u);
        struct set<T>* s = new set<T>{junkval};
        insert(adj_dict, u, s);
      }
    }

    void add_edge(T u, T v, bool add = false) {
      if (!add && !(has_vertex(u) && has_vertex(v))) {
        return;
      }
      add_vertex(u);
      add_vertex(v);
      if (!has_edge(u, v)){
        insert(*adj_dict[u], v);
        insert(*adj_dict[v], u);
        insert(edges, edge<T>{u, v});
      }
    }

    void remove_vertex(T u){
      return;
    }

    void remove_edge(T u, T v){
      return;
    }

    bool has_vertex(T u){
      return mem(V, u);
    }

    bool has_edge(T u, T v){
      return mem(*adj_dict[u], v);
    }

    struct set<T>& operator[](T u){
      return *adj_dict[u];
    }

    int num_vertices() {
      return V.size;
    }

    int num_edges(){
      return edges.size;
    }

    int to_igraph(struct IGraph& IG){
      // if (IG.adj_list != NULL){
      //   free_igraph(&IG);
      // }
      init_graph(&IG, num_vertices());
      int i{0};
      for (T u : V){
        insert(this->imapping, i, u);
        insert(this->reverse_mapping, u, i);
        std::cout << "u: " << u << ", i: " << i << ", map[i]: " << this->imapping[i] << ", rmap[u]: " << this->reverse_mapping[u] << "\n";
        i++;
      }
      for (struct edge<T> e: edges){
        std::cout << e << "\n";
        std::cout << this->reverse_mapping[e.u] << " " << this->reverse_mapping[e.v] << "\n";
        ::add_edge(&IG, this->reverse_mapping[e.u], this->reverse_mapping[e.v]);
      }
      return 0;
    } 
};

template<typename T>
std::ostream& operator<<(std::ostream& os, edge<T>& e) {
    os << "{" << e.u << ", " << e.v << "}";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, Graph<T>& G) {
    os << "𝑉 = " << G.vertices();
    os << "𝐸 = " << G.get_edges();
    return os;
}

#endif //CPP_GRAPH_H
