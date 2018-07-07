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


const CAP: usize = 50;

#[derive(Deserialize)]
struct SparseVector {
    #[serde(rename = "pid")]
    id: String,
    #[serde(rename = "features")]
    pos: Vec<u32>,
    #[serde(rename = "scores")]
    val: Vec<f32>,
}

#[derive(Clone, PartialEq, Eq)]
struct Cosine {
    id: String,
    val: float_ord::FloatOrd<f32>,
}

impl PartialOrd for Cosine {
    fn partial_cmp(&self, other: &Cosine) -> Option<Ordering> {
        other.val.partial_cmp(&self.val)
    }
}

impl Ord for Cosine {
    fn cmp(&self, other: &Cosine) -> Ordering {
        other.val.cmp(&self.val)
    }
}

#[derive(Serialize)]
struct CosineClean {
    #[serde(rename = "pid")]
    id: String,
    sim: f32,
}

#[derive(Serialize)]
struct Similar {
    #[serde(rename = "pid")]
    id: String,
    similar: Vec<CosineClean>,
}



fn main() {
    let path = args().nth(1).unwrap();
    let chunk: usize = args().nth(2).unwrap().parse().unwrap();
    // let nchunks: u32 = args().nth(3).unwrap().parse().unwrap();
    let file = File::open(&path).unwrap();

    eprintln!("[{}] START {}", Local::now().format("%Y-%m-%d %H:%M:%S").to_string(), &path);

    let products: Vec<SparseVector> =
        BufReader::new(file).lines()
        .map(|line| line.unwrap())
        .map(|line| serde_json::from_str(&line) as Result<SparseVector, serde_json::Error>)
        .map(|pbow| pbow.unwrap())
        .collect();
    
    eprintln!("[{}] INTAKE COMPLETED", Local::now().format("%Y-%m-%d %H:%M:%S").to_string());

    let total: usize = products[0..chunk].par_iter()
        .map(|prod| topn(prod, &products))
        .map(|sims| serde_json::to_string(&sims).unwrap())
        .map(|strn| {
            println!("{}", strn);
            1
        })
        .sum();
    
    eprintln!(
        "[{}] DONE IN {} PASSES OVER {} PRODUCTS, TOTALING {} OPERATIONS",
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        &total,
        products.len(),
        &total*products.len(),
    );
}


fn topn(reference: &SparseVector, products: &Vec<SparseVector>) -> Similar {
    let mut heap = BinaryHeap::from(vec![Cosine{id: "--".to_owned(), val: float_ord::FloatOrd(0.)}; CAP]);

    for prod in products {
        let cos = if reference.pos.len() <= prod.pos.len() { cosine(reference, prod) } else { cosine(prod, reference) };
        if cos >= heap.peek().unwrap().val.0 && prod.id != reference.id {
            heap.pop();
            heap.push(Cosine{
                id: prod.id.clone(),
                val: float_ord::FloatOrd(cos),
            });
        }
    }

    let ub: Vec<CosineClean> = heap.into_sorted_vec().into_iter()
        .filter(|cos| cos.id != "--")
        .map(|cos| CosineClean{id: cos.id, sim: cos.val.0})
        .collect();

    Similar{id: reference.id.clone(), similar: ub}
}


fn cosine(reference: &SparseVector, comparison: &SparseVector) -> f32 {
    let numerator =
        reference.pos.iter()
        .enumerate()
        .map(|(refidx, feature)| {
            match comparison.pos.binary_search(&feature) {
                Ok(cmpidx) => reference.val[refidx] * comparison.val[cmpidx],
                _ => 0 as f32,
            }
        })
        .sum();

    numerator
}
