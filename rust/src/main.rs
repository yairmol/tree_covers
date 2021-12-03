mod graph;
mod cgraph;

use cgraph::{IGraph, path_graph, my_vec_to_vec};
use std::ptr;
// use graph::Graph;
// use std::collections::HashSet;

fn main() {
    // let g = Graph::<i32>::path_graph(10);
    // println!("G: {:}", g);
    // let mut s = HashSet::<i32>::new();
    // s.insert(1);
    // s.insert(2);
    // s.insert(5);
    // let gs = g.induced_subgraph(&s);
    // println!("G[S]: {}", gs);
    let mut g = IGraph {
        num_vertices: 0,
        num_edges: 0,
        adj_list: ptr::null_mut()
    };
    unsafe {
        path_graph(&mut g, 10);
        // *(*g.adj_list.offset(5)).arr.offset(1)
        println!("{} {} {:?}", g.num_vertices, g.num_edges, my_vec_to_vec(&(*g.adj_list)));
        let g2 = g.to_graph();
        println!("{}", g2);
    }
}