// mod crate::cgraph;

use std::collections::HashSet;
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt;

use crate::cgraph;
use cgraph::IGraph;


#[derive(Hash)]
#[derive(PartialEq)]
#[derive(Eq)]
#[derive(PartialOrd)]
#[derive(Debug)]
pub struct Edge<T: Hash + PartialOrd + Eq> {
    pub u: T,
    pub v: T
}

impl<T: Hash + PartialOrd + Eq> Edge<T> {
    pub fn new(u: T, v: T) -> Edge<T>{
        Edge {u, v}
    }
}

impl<T: PartialOrd + Eq + Hash + fmt::Debug> fmt::Display for Edge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.u, self.v)
    }

}

pub struct Graph<T: PartialOrd + Eq + Hash + Copy> {
    adj_dict: HashMap<T, HashSet<T>>,
    edges: HashSet<Edge<T>>,
    changed: bool,
    imapping: HashMap<i32, T>,
    reverse_mapping: HashMap<T, i32>
}

impl<T: PartialOrd + Eq + Hash + Copy> Graph<T> {
    pub fn new() -> Graph<T>{
        Graph {
            adj_dict: HashMap::new(),
            edges: HashSet::new(),
            changed: true,
            imapping: HashMap::new(),
            reverse_mapping: HashMap::new(),
        }
    }

    

    pub fn path_graph(n: i32) -> Graph<i32> {
        let mut G = Graph::<i32>::new();
        G.add_vertex(0);
        for u in 1..n {
            G.add_vertex(u);
            G.add_edge(u - 1, u);
        }
        G
    }

    pub fn adj_dict(&self) -> &HashMap<T, HashSet<T>> {
        &self.adj_dict
    }

    pub fn edges(&self) -> &HashSet<Edge<T>>{
        &self.edges
    }

    pub fn add_vertex(&mut self, u: T) -> (){
        if !self.adj_dict.contains_key(&u) {
            self.adj_dict.insert(u, HashSet::new());
            self.changed = true;
        }
    }

    pub fn add_edge(&mut self, u: T, v: T) {
        if !(self.has_vertex(&u) && self.has_vertex(&v)) {
            false;
        }
        if !self.has_edge(&u, &v) {
            match self.adj_dict.get_mut(&u) {
                Some(s) => s.insert(v),
                None => panic!()
            };
            match self.adj_dict.get_mut(&v) {
                Some(s) => s.insert(u),
                None => panic!()
            };
            self.edges.insert(Edge::<T>::new(u, v));
        }
    }

    fn remove_vertex(&mut self, u: &T){
        self.adj_dict.remove(u);
    }

    fn remove_edge(&mut self, u: &T, v: &T){
      return;
    }

    pub fn has_vertex(&self, u: &T) -> bool{
        self.adj_dict.contains_key(u)
    }

    pub fn has_edge(&self, u: &T, v: &T) -> bool{
        self.adj_dict[u].contains(v)
    }

    fn set_mapping(&mut self){
        if self.changed {
            let mut i = 0;
            for u in self.adj_dict.keys() {
                self.imapping.insert(i, *u);
                self.reverse_mapping.insert(*u, i);
                i = i + 1;
            }
            self.changed = false;
        }
    }

    pub fn to_igraph(&mut self, IG: &mut IGraph){
        unsafe {
            cgraph::init_graph(IG, self.adj_dict.len() as i32);
        }
        self.set_mapping();
        unsafe {
            for e in &self.edges {
                cgraph::add_edge(IG, self.reverse_mapping[&e.u], self.reverse_mapping[&e.v]);
            }
        }
    }

    pub fn induced_subgraph(&self, s: &HashSet<T>) -> Graph<T>{
        let mut gs = Graph::new();
        for u in s.iter() {
            gs.add_vertex(*u);
        }
        for e in &self.edges {
            if s.contains(&e.u) && s.contains(&e.v) {
                gs.add_edge(e.u, e.v);
            }
        }
        gs
    }

}

impl<T: PartialOrd + Eq + Hash + Copy + fmt::Debug> fmt::Display for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.adj_dict.keys(), self.edges())
    }

}