#define GRAPH_VEC
extern "C" {
    #include "graph/include/graph.h"
}
#include <iostream>
#include "cpp/include/graph.h"
#include "cpp/include/stack.h"
#include "cpp/include/algorithms.h"
// #include "cpp/include/algorithms.h"

void path_graph2(Graph<int>& G, int n){
    for (size_t i = 0; i < n - 1; i++) {
        G.add_edge(i, i + 1, true);
    }
}

int main(){
    Graph<int> G{-1};
    std::cout << G;;
    // path_graph2(G, 10);
    G.add_edge(1, 2, true);
    G.add_edge(3, 4, true);
    G.add_edge(5, 6, true);
    G.add_edge(5, 7, true);
    G.add_edge(6, 7, true);
    std::cout << G;;
    struct IGraph IG; G.to_igraph(IG);
    // struct IGraph IG;
    // path_graph(&IG, 10);
    print_graph(&IG);
    struct set<int> s{-1};
    insert(s, 1);
    insert(s, 4);
    Graph<int>* sub = G[s];
    std::cout << *sub;
    std::cout << get(s) << "\n";
    Graph<int> Tree{-1};
    dfs_tree(G, Tree);
    std::cout << Tree;
}