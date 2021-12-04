// mod cgraph;
use std::slice;
// use libc;
// use std::ffi;
use crate::graph;
use graph::{Graph};
use std::ptr;

#[repr(C)]
#[derive(Clone)]
pub struct Vector {
    pub arr: *mut i32,
    pub cur: i32,
    pub current_size: i32
}

#[repr(C)]
pub struct IGraph {
    pub num_vertices: i32,
    pub num_edges: i32,
    pub adj_list: *mut Vector
}

type Metric = extern fn(i32, i32) -> f32;

#[link(name = "graph")]
extern "C" {
    pub fn init_graph(G: *mut IGraph, n: i32);
    pub fn path_graph(G: *mut IGraph, n: i32);
    pub fn add_edge(G: *mut IGraph, u: i32, v: i32);
    pub fn remove_edge(G: *mut IGraph, u: i32, v: i32);
    pub fn shortest_path_length(G: *mut IGraph, s: i32, t: i32, cutoff: i32) -> i32;
    pub fn embedding_distortion(X: *mut i32, size: usize, d1: Metric, d2: Metric);
    pub fn single_source_shortest_path(G: *const IGraph, s: i32) -> *mut i32;
}


pub fn my_vec_to_vec(myv: &Vector) -> Vec<i32> {  
    unsafe {
        let v = slice::from_raw_parts(myv.arr, myv.cur as usize).to_vec();
        // libc::free(myv.arr as *mut ffi::c_void);
        v
    }
}

impl IGraph {
    pub fn new() -> IGraph{
        IGraph {
            num_vertices: 0,
            num_edges: 0,
            adj_list: ptr::null_mut()
        }
    }
    pub fn to_graph(&mut self) -> Graph<i32> {
        let mut g = Graph::<i32>::new();
        for i in 0..self.num_vertices {
            g.add_vertex(i);
        }
        unsafe {
            // let adj_list = slice::from_raw_parts(self.adj_list, self.num_vertices as usize).to_vec();
            for u in 0..self.num_vertices {
                let neighbors = my_vec_to_vec(&*self.adj_list.offset(u as isize));
                for v in neighbors {
                    g.add_edge(u, v);
                }
            }
        }
        g
    }
}


// #[repr(C)]
// #[derive(Debug)]
// struct MyStruct {
//     x: *mut i32,
//     y: i32,
//     z: i32,
// }

// extern "C" {
//     pub fn myfunc(x: i32, y: i32, z: i32) -> *mut MyStruct;
// }
