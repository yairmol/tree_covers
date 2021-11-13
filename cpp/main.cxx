// A simple program that computes the square root of a number
#include <cmath>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include "Graphs.h"
#include "Timer.h"
#include "GraphAlgorithms.cxx"
#include "include/json.hpp"

using json = nlohmann::json;

template<typename T>
void print_graph(Graph<T> g){
  std::cout << "V = {";
  for (auto &u : g.get_vertices()){
    std::cout << u << ", ";
  }
  std::cout << "}" << std::endl;
  std::cout << "E = {";
  for (auto e : g.get_edges()){
    std::cout << "{" << e.u << ", " << e.v << "}, ";
  }
  // for (auto &adj : g.get_adj_list()){
  //   T u{adj.first};
  //   for (auto &v : adj.second){
  //     std::cout << "{" << u << ", " << v << "}, ";
  //   }
  // }
  std::cout << "}" << std::endl;
}

template<typename K, typename V>
void print_map(std::map<K, V> m, void (*print_key)(K), void (*print_value)(V))
{
  std::cout << "{ ";
  for (auto &item : m) {
    print_key(item.first);
    std::cout << ": ";
    print_value(item.second);
    std::cout << ", ";
  }
  std::cout << "}\n";
}

template<typename T>
void print_vector(std::vector<T> x){
  std::cout << "[";
  for (auto &item : x){
    std::cout << item << ", ";
  }
  std::cout << "]" << std::endl;
}

void add_pair(std::map<int, std::string> &my_map, int key, std::string value){
  my_map.insert(std::pair<int, std::string> {key, value});
}

// void map_expireiemnting(){
//   std::map<int, std::string> map1 = {
//     {1, "Apple",},
//     {2, "Banana",},
//     {3, "Mango",},
//     {4, "Raspberry",},
//     {5, "Blackberry",},
//     {6, "Cocoa",}
//   };
//   print_map(map1);
//   std::cout << std::endl;
//   add_pair(map1, 7, "hello world");
//   add_pair(map1, 8, "2");
//   add_pair(map1, 9, "3");
//   print_map(map1);
// }

// void set_vs_vector_erase(){
//   int size {1000 * 1000 * 10};
//   Timer t;
//   // std::vector<int> x{};
//   std::set<int> x{};
//   for (int i(0); i < size; i++){
//     // x.push_back(i);
//     x.insert(i);
//   }
//   double first_elapsed{t.elapsed()};
//   std::cout << "constructing the set " << first_elapsed << " size " << x.size() << std::endl;
//   t.reset();
//   // print_vector(x);
//   x.erase(std::remove(x.begin(), x.end(), size - 2), x.end());
//   x.erase(size - 2);
//   double second_elapsed {t.elapsed()};
//   std::cout << "erasing " << second_elapsed << " size " << x.size() << std::endl;
//   // print_vector(x);
// }

void graph_expiriementing(){
  Graph<int> g {};
  for (int i{0}; i < 3; i++){
    g.add_vertex(i);
  }
  g.add_edge(1, 2);
  g.add_edge(0, 2);
  g.add_edge(0, 3);
  print_graph(g);
}

void check_copy_constructor(){
  std::vector<int> x{1, 2, 3};
  std::vector<int> y{x};
  print_vector(x);
  print_vector(y);
  x.push_back(5);
  print_vector(x);
  print_vector(y);
}

void print_int(int x){
  std::cout << x;
}

void check_map_init(){
  std::map<int, std::vector<int> > mymap{};
  mymap[1] = std::vector<int>{1, 2, 3};
  print_map(mymap, print_int, print_vector);
}

void print_paths_map(std::map<int, path_t<int>> paths_map){
    print_map(paths_map, print_int, print_vector);
};

void check_bfs(){
  Graph<int> g {};
  for (int i{0}; i < 3; i++){
    g.add_vertex(i);
  }
  g.add_edge(1, 2);
  g.add_edge(0, 2);
  std::map<int, std::map<int, path_t<int>>> paths = all_pairs_shortest_paths(g);
  json paths_json = paths;
  std::cout << paths_json.dump() << std::endl;
   auto print_paths_map = [](std::map<int, path_t<int>> paths_map){
     print_map(paths_map, print_int, print_vector);
   };
   print_map(paths, print_int, print_paths_map);
}

template <typename T>
void print_edge(Edge<T> e){
  std::cout << "{" << e.u << ", " << e.v << "}";
}

template <typename T>
void print_set(std::set<T>& s, void (*print_element)(T)){
  std::cout << "{";
  for (auto& e: s){
    print_element(e);
    std::cout << ", ";
  }
  std::cout << "}" << std::endl;
}

void set_iterator_performance(){
  Timer t;
  std::set<int> s;
  for (int i{0}; i < 1000 * 1000; i++){
    s.insert(i);
  }
  double time = t.elapsed();
  std::cout << "Set build in " << time << " seconds. set size is " << s.size() << std::endl;
  int j{0};
  t.reset();
  for (int i : s){
    j++;
  }
  time = t.elapsed();
  std::cout << "Set iterated in " << time << " seconds" << std::endl;
  t.reset();
  for (int i{0}; i < 1000 * 1000; i++){
    s.find(i);
  }
  time = t.elapsed();
  std::cout << "found all set items in " << time << " seconds" << std::endl;
}

void ssps_performance(){
  Timer t;
  Graph<int> g = diamond_graph(8);
  double time = t.elapsed();
  std::cout << "Graph build Time: " << time << std::endl;
  t.reset();
  single_source_shortest_paths_length(g, 1);
  time = t.elapsed();
  std::cout << "Time sssp: " << time << std::endl;
}

template <typename T>
void print_vertex_pointer(Vertex<T>* v){
  std::cout << v->value;
}

template <typename T>
void print_vertex(Vertex<T>& v){
  std::cout << v.value;
}

void pointers_check(){
  Vertex<int> u {1, false};
  Vertex<int> v {1, false};
  std::set<Vertex<int>> s;
  s.insert(u);
  s.insert(v);
  print_set(s, print_vertex);
}

int main(int argc, char* argv[])
{
  pointers_check();
  return 0;
}