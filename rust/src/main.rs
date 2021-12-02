mod graph;

use graph::Graph;
use std::collections::HashSet;

fn main() {
    let g = Graph::<i32>::path_graph(10);
    println!("G: {:}", g);
    let mut s = HashSet::<i32>::new();
    s.insert(1);
    s.insert(2);
    s.insert(5);
    let gs = g.induced_subgraph(&s);
    println!("G[S]: {}", gs);
}