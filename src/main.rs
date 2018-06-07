#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate serde;

use std::env::args;
use std::io::{BufReader,BufRead};
use std::fs::File;
use serde_json::Error;


#[derive(Serialize, Deserialize)]
struct ProductInput {
    pid: String,
    feats: Vec<u32>,
    scores: Vec<f32>,
}


#[derive(Serialize, Deserialize)]
struct ProductLight {
    feats: Vec<u32>,
    scores: Vec<f32>,
}


fn main() {
    let mut argv = args();
    let path = argv.next().unwrap();
    let file = File::open(path).unwrap();

    for line in BufReader::new(file).lines() {
        println!("{}", line.unwrap());
    }
}