extern "C" {
    #include "graph/include/graph.h"
}
#include <iostream>
#include "cpp/include/graph.h"

void path_graph2(Graph<int>& G, int n){
    for (size_t i = 0; i < n - 1; i++) {
        G.add_edge(i, i + 1, true);
    }
}

int main(){
    Graph<int> G{-1};
    std::cout << G;;
    path_graph2(G, 10);
    std::cout << G;;
    struct IGraph IG; G.to_igraph(IG);
    // struct IGraph IG;
    // path_graph(&IG, 10);
    print_graph(&IG);
}