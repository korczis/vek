#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate float_ord;
extern crate rayon;
extern crate chrono;

use std::env::args;
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use rayon::prelude::*;
use chrono::prelude::*;

mod vecs;
use vecs::{Distances, SparseVector};


const CAP: usize = 50;

#[derive(Serialize, Deserialize)]
struct Wrap {
    #[serde(rename = "pid")]
    id: String,
    #[serde(flatten)]
    vector: SparseVector,
}

#[derive(Clone, PartialEq, Eq)]
struct Cmp {
    id: String,
    val: float_ord::FloatOrd<f32>,
}

impl PartialOrd for Cmp {
    fn partial_cmp(&self, other: &Cmp) -> Option<Ordering> {
        other.val.partial_cmp(&self.val)
    }
}

impl Ord for Cmp {
    fn cmp(&self, other: &Cmp) -> Ordering {
        other.val.cmp(&self.val)
    }
}


#[derive(Serialize)]
struct CmpClean {
    #[serde(rename = "pid")]
    id: String,
    sim: f32,
}

#[derive(Serialize)]
struct Similar {
    #[serde(rename = "pid")]
    id: String,
    similar: Vec<CmpClean>,
}



fn main() {
    let path = args().nth(1).unwrap();
    let chunk: usize = args().nth(2).unwrap().parse().unwrap();
    // let nchunks: u32 = args().nth(3).unwrap().parse().unwrap();
    let file = File::open(&path).unwrap();

    eprintln!("[{}] START {}", Local::now().format("%Y-%m-%d %H:%M:%S").to_string(), &path);

    let products: Vec<Wrap> =
        BufReader::new(file).lines()
        .map(|line| line.unwrap())
        .map(|line| serde_json::from_str(&line) as Result<Wrap, serde_json::Error>)
        .map(|pbow| pbow.unwrap())
        .collect();
    
    eprintln!("[{}] INTAKE COMPLETED", Local::now().format("%Y-%m-%d %H:%M:%S").to_string());

    let total: usize = products[0..chunk].par_iter()
        .map(|prod| topn(prod, &products))
        .map(|sims| serde_json::to_string(&sims).unwrap())
        .map(|strn| println!("{}", strn))
        .count();
    
    eprintln!(
        "[{}] DONE IN {} PASSES OVER {} PRODUCTS, TOTALING {} OPERATIONS",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        &total,
        products.len(),
        &total*products.len(),
    );
}


fn topn(reference: &Wrap, products: &Vec<Wrap>) -> Similar {
    let mut heap = BinaryHeap::from(vec![Cmp{id: "--".to_owned(), val: float_ord::FloatOrd(0.)}; CAP]);

    for prod in products {
        let cos = reference.vector.cosine(&prod.vector);
        if cos >= heap.peek().unwrap().val.0 && prod.id != reference.id {
            heap.pop();
            heap.push(Cmp{
                id: prod.id.clone(),
                val: float_ord::FloatOrd(cos),
            });
        }
    }

    let ub: Vec<CmpClean> = heap.into_sorted_vec().into_iter()
        .filter(|cos| cos.id != "--")
        .map(|cos| CmpClean{id: cos.id, sim: cos.val.0})
        .collect();

    Similar{id: reference.id.clone(), similar: ub}
}
