use crate::graph::{Graph};
use rand::prelude::*;

pub fn fast_gnp_random_graph(n: i32, p: f64) -> Graph<i32> {
    let mut g = Graph::new();
    let mut rng = rand::thread_rng();
    for u in 0..n {
        g.add_vertex(u);
    }
    let mut w = -1;
    let lp = (1.0 - p).log(10.);
    let mut v: i32 = 1;

    while v < n {
        let r: f64 = rng.gen();
        let lr = (1.0 - r).log(10.);
        w = w + 1 + ((lr / lp) as i32);
        while w >= v && v < n {
            w = w - v;
            v = v + 1;
        }
        if v < n {
            g.add_edge(v, w);
        }
    }
    g
}