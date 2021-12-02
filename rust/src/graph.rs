use std::collections::HashSet;
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt;

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

impl<'a, T: PartialOrd + Eq + Hash + fmt::Debug> fmt::Display for Edge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.u, self.v)
    }

}

pub struct Graph<'a, T: PartialOrd + Eq + Hash + Copy> {
    adj_dict: HashMap<T, HashSet<T>>,
    edges: HashSet<Edge<T>>,
    changed: bool,
    imapping: HashMap<i32, &'a T>,
    reverse_mapping: HashMap<&'a T, i32>
}

impl<'a, T: PartialOrd + Eq + Hash + Copy> Graph<'a, T> {
    pub fn new() -> Graph<'a, T>{
        Graph {
            adj_dict: HashMap::new(),
            edges: HashSet::new(),
            changed: true,
            imapping: HashMap::new(),
            reverse_mapping: HashMap::new(),
        }
    }

    

    pub fn path_graph<'b>(n: i32) -> Graph<'b, i32> {
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

    // /**
    //  * @brief creates a mapping from the graph vertices to
    //  * {0, ... , n}, where n = num_vertices()
    //  */
    // void set_mapping(){
    //   if (changed){
    //     int i{0};
    //     for (T u : V){
    //       insert(this->imapping, i, u);
    //       insert(this->reverse_mapping, u, i);
    //       // std::cout << "u: " << u << ", i: " << i << ", map[i]: " << this->imapping[i] << ", rmap[u]: " << this->reverse_mapping[u] << "\n";
    //       i++;
    //     }
    //     changed = false;
    //   }
    // }

    // int to_igraph(struct IGraph& IG){
    //   // if (IG.adj_list != NULL){
    //   //   free_igraph(&IG);
    //   // }
    //   init_graph(&IG, num_vertices());
    //   set_mapping();
    //   for (struct edge<T> e: edges){
    //     ::add_edge(&IG, this->reverse_mapping[e.u], this->reverse_mapping[e.v]);
    //   }
    //   return 0;
    // }

    pub fn induced_subgraph(&self, s: &HashSet<T>) -> Graph<'a, T>{
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

impl<'a, T: PartialOrd + Eq + Hash + Copy + fmt::Debug> fmt::Display for Graph<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.adj_dict.keys(), self.edges())
    }

}