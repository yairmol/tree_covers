#![allow(dead_code)]

mod graph;
mod cgraph;
mod algorithms;
mod generators;
mod embeddings;

// use cgraph::{IGraph, path_graph, my_vec_to_vec};
use cgraph::IGraph;
// use std::ptr;
use graph::Graph;
use generators::fast_gnp_random_graph;
use std::collections::HashSet;

pub fn make_f() -> impl FnMut() -> f32 {
    let mut current = 0.;
    move || {
        current += 1.;
        current
    }
}


pub fn path_distortion_check() {
    unsafe {
        let mut g = IGraph::new();
        cgraph::path_graph(&mut g, 10);
        cgraph::add_edge(&mut g, 0, 2);
        let mut h = IGraph::new();
        cgraph::path_graph(&mut h, 10);
        // cgraph::remove_edge(&mut h, 1, 2);
        // *(*g.adj_list.offset(5)).arr.offset(1)
        let mut dg = algorithms::make_graph_metric(g);
        for u in 0..9 {
            for v in 0..9 {
                // print!("d({}, {}) = {}; ", u, v, dg(u, v));
                print!("{} ", dg(&u, &v));
            }
            println!();
        }
        println!();
        let mut dh = algorithms::make_graph_metric(h);
        for u in 0..9 {
            for v in 0..9 {
                // print!("d({}, {}) = {}; ", u, v, dg(u, v));
                print!("{} ", dh(&u, &v));
            }
            println!();
        }
        let points: Vec<i32> = (0..10).collect();
        println!("{:?}", points);
        let distortion = embeddings::embedding_distortion(&points, &mut dg, &mut dh);
        println!("distortion: {}", distortion);
        
    }
}

fn induced_subgraph_check() {
    let g = Graph::<i32>::path_graph(10);
    println!("G: {:}", g);
    let mut s = HashSet::<i32>::new();
    s.insert(1);
    s.insert(2);
    s.insert(5);
    let gs = g.induced_subgraph(&s);
    println!("G[S]: {}", gs);
}


fn check_myvec_to_vec() {
    // let mut g = IGraph {
    //     num_vertices: 0,
    //     num_edges: 0,
    //     adj_list: ptr::null_mut()
    // };
    // println!("{} {} {:?}", g.num_vertices, g.num_edges, my_vec_to_vec(&(*g.adj_list)));
    //     let g2 = g.to_graph();
    //     println!("{}", g2);
}

fn check_low_stretch_spanner() {
    let x = 1 << 14;
    println!("{}", x);
    let mut g = fast_gnp_random_graph(x, 0.001);
    println!("Graph with {} vertices and {} edges. connected: {}", g.adj_dict().len(), g.edges().len(), g.is_connected());
    if g.is_connected() {
        let stretch = (2. * (g.adj_dict().len() as f64).log(2.)) + 1.;
        println!("strecth {}", stretch);
        let mut h = algorithms::low_stretch_spanner(&mut g, stretch.floor() as usize, ((x as f64) * 1.5) as usize);
        println!("Graph with {} vertices and {} edges", h.adj_dict().len(), h.edges().len());
        let mut  ig = IGraph::new();
        g.to_igraph(&mut ig);
        let mut ih = IGraph::new();
        h.to_igraph(&mut ih);
        let mut dg = algorithms::make_graph_metric(ig);
        let mut dh = algorithms::make_graph_metric(ih);
        let points: Vec<i32> = (0..x).collect();
        let distortion = embeddings::embedding_distortion(&points, &mut dg, &mut dh);
        println!("stretch: {}", distortion);
    }
}


fn main() {
    check_low_stretch_spanner();
}