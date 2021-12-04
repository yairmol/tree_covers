// mod crate::cgraph;

use std::collections::{VecDeque, HashSet, HashMap};
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
    pub imapping: HashMap<i32, T>,
    pub reverse_mapping: HashMap<T, i32>
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
        let mut g = Graph::<i32>::new();
        g.add_vertex(0);
        for u in 1..n {
            g.add_vertex(u);
            g.add_edge(u - 1, u);
        }
        g
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
                Some(s) => {s.insert(v);},
                None => ()
            };
            match self.adj_dict.get_mut(&v) {
                Some(s) => {s.insert(u);},
                None => ()
            };
            self.edges.insert(Edge::<T>::new(u, v));
        }
    }

    fn remove_vertex(&mut self, u: &T){
        self.adj_dict.remove(u);
        self.changed = true;
    }

    fn remove_edge(&mut self, u: &T, v: &T){
        match self.adj_dict.get_mut(u) {
            Some(s) => {s.remove(v);}
            None => (),
        };
        match self.adj_dict.get_mut(v) {
            Some(s) => {s.remove(u);}
            None => (),
        };
        self.edges.remove(&Edge::new(*u, *v));
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

    pub fn to_igraph(&mut self, ig: &mut IGraph){
        unsafe {
            cgraph::init_graph(ig, self.adj_dict.len() as i32);
        }
        self.set_mapping();
        unsafe {
            for e in &self.edges {
                cgraph::add_edge(ig, self.reverse_mapping[&e.u], self.reverse_mapping[&e.v]);
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

    pub fn is_connected(&self) -> bool{
        let mut found = HashSet::<&T>::new();
        let mut q = VecDeque::<&T>::new();
        let s = match self.adj_dict.keys().next() {
            Some(s) => s,
            None => return true
        };
        q.push_back(s);
        while !q.is_empty() {
            match q.pop_front() {
                Some(u) => {
                    found.insert(u);
                    for v in &self.adj_dict[u] {
                        if !found.contains(v) {
                            q.push_back(v);
                        }
                    }
                },
                None => ()
            }
        }
        found.len() == self.adj_dict.len()
    }

}

impl<T: PartialOrd + Eq + Hash + Copy + fmt::Debug> fmt::Display for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.adj_dict.keys(), self.edges())
    }

}