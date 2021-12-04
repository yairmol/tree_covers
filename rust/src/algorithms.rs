use crate::graph::{Graph};
use crate::cgraph::{IGraph};
use crate::cgraph;

use std::hash::{Hash};
use std::ops::FnMut;
use libc;
use std::ffi;


pub fn low_stretch_spanner<T: PartialOrd + Eq + Hash + Copy>(g: &mut Graph<T>, k: usize, max_edges: usize) -> Graph<T> {
    println!("k: {} max edges: {}", k, max_edges);
    let mut h = Graph::<T>::new();
    for u in g.adj_dict().keys() {
        h.add_vertex(*u);
    }
    let mut ih = IGraph::new();
    h.to_igraph(&mut ih);
    for e in g.edges() {
        if h.edges().len() >= max_edges {
            break;
        }
        unsafe {
            let iu = h.reverse_mapping[&e.u];
            let iv = h.reverse_mapping[&e.v];
            let d = cgraph::shortest_path_length(&mut ih, iu, iv, (k - 1) as i32);
            if d == -1 {
                h.add_edge(e.u, e.v);
                cgraph::add_edge(&mut ih, iu, iv);
            }
        }
    }
    h
}


pub fn make_graph_metric(ig: IGraph) -> impl FnMut(&i32, &i32) -> f32 {
    let mut current = 0;
    let mut d = unsafe {
        cgraph::single_source_shortest_path(&ig, current)
    };
    move |u, v| {
        if current == *u {
            unsafe {
                (*d.offset(*v as isize)) as f32
            }
        } else {
            current = *u;
            unsafe {
                libc::free(d as *mut ffi::c_void);
            }
            d = unsafe {
                cgraph::single_source_shortest_path(&ig, current)
            };
            unsafe {
                (*d.offset(*v as isize)) as f32
            }
        }
    }
}