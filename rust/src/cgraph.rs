// mod cgraph;
use std::slice;
use libc;
use std::ffi;
use crate::graph;
use graph::{Graph};

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


#[link(name = "graph")]
extern "C" {
    pub fn init_graph(G: *mut IGraph, n: i32);
    pub fn path_graph(G: *mut IGraph, n: i32);
    pub fn add_edge(G: *mut IGraph, u: i32, v: i32);
}


pub fn my_vec_to_vec(myv: &Vector) -> Vec<i32> {  
    unsafe {
        let v = slice::from_raw_parts(myv.arr, myv.cur as usize).to_vec();
        // libc::free(myv.arr as *mut ffi::c_void);
        v
    }
}

impl IGraph {
    pub fn to_graph(&mut self) -> Graph<i32> {
        let mut G = Graph::<i32>::new();
        for i in 0..self.num_vertices {
            G.add_vertex(i);
        }
        unsafe {
            // let adj_list = slice::from_raw_parts(self.adj_list, self.num_vertices as usize).to_vec();
            for u in 0..self.num_vertices {
                let neighbors = my_vec_to_vec(&*self.adj_list.offset(u as isize));
                for v in neighbors {
                    G.add_edge(u, v);
                }
            }
        }
        G
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
