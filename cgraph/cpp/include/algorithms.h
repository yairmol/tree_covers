extern "C" {
    #include "../../graph/include/graph.h"
}
#include "graph.h"
#include "hash_dict.h"
#include "stack.h"


/**
 * @brief return -1 if all indices are visited
 * otherwise return an index which wasn't visited
 * 
 * @param visited 
 * @param size 
 * @return int 
 */
int not_visited(bool* visited, int size){
    for (size_t i = 0; i < size; i++) {
        if (!visited[i]){
            return i;
        }
    }
    return -1;
}


void dfs_tree_rec(struct IGraph& IG, struct IGraph& T, int s, bool* visited){
    for (int i = 0; i < IG.adj_list[s].cur; i++){
        int v = IG.adj_list[s].arr[i];
        if (!visited[v]){
            add_edge(&T, s, v);
            visited[v] = true;
            dfs_tree_rec(IG, T, v, visited);
        }
    }
}

template <typename T>
int dfs_tree(Graph<T>& G, Graph<T>& tree) {
    if (G.num_vertices() == 0){
        return 0;
    }
    struct IGraph IG;
    G.to_igraph(IG);
    struct IGraph IT;
    init_graph(&IT, IG.num_vertices);
    bool visited[IG.num_vertices];
    int u;
    while ((u = not_visited(visited, IG.num_vertices)) != -1) {
        dfs_tree_rec(IG, IT, u, visited);
    }
    for (int u = 0; u < IT.num_vertices; u++){
        for (int i = 0; i < IT.adj_list[u].cur; i++){
            int v = IT.adj_list[u].arr[i];
            if (u < v){
                tree.add_edge(G.imapping[u], G.imapping[v], true);
            }
        }
    }
    return 0;
}

// template <typename T>
// void connected_components(Graph<T>& G, set<set<T>*>& CCs){

// }