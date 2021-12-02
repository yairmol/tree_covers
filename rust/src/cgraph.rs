// mod cgraph;

#[repr(C)]
#[derive(Debug)]
struct MyStruct {
    x: *mut i32,
    y: i32,
    z: i32,
}

#[link(name = "graph")]
extern "C" {
    fn myfunc(x: i32, y: i32, z: i32) -> *mut MyStruct;
}

fn main() {
    // let mut g = Graph::new();
    // g.add_vertex(1);
    // g.add_vertex(2);
    // for v in g.adj_list().keys(){
    //     println!("v: {}", v);
    // }
    // g.add_edge(&1, &2);
    // println!("edges: {:}", g);
    unsafe {
        let x: *mut MyStruct = myfunc(5, 2, 3);
        println!("Absolute value of -3 according to C: {:?}", *x);
    }
}
