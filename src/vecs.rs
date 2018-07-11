extern crate serde;
extern crate serde_json;
use std::cmp::Ordering;

// index MUST be ordered and not contain repeated elements
// index.len() MUST be equal to value.len()
// it also assumes the vector has AT LEAST 1 dimension != 0
#[derive(Serialize, Deserialize)]
pub struct SparseVector {
    #[serde(rename = "features")]
    pub index: Vec<u32>,
    #[serde(rename = "scores")]
    pub value: Vec<f32>,
}


// ALL vector comparisons must be between vectors of equal value.len() != 0
pub struct DenseVector {
    pub value: Vec<f32>,
}


pub trait Distances {
    fn norm(&self) -> f32;

    // fn mae(&self, other: &Self, n: usize) -> f32;
    fn sad(&self, other: &Self) -> f32;

    // fn mse(&self, other: &Self, n: usize) -> f32;
    fn euclidean(&self, other: &Self) -> f32;
    fn ssd(&self, other: &Self) -> f32;

    fn cosine(&self, other: &Self) -> f32;  // assumes vectors are L2-normalised
}


pub trait Transformations {
    fn normalize(&self) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
}


impl Distances for SparseVector {
    fn norm(&self) -> f32 { self.value.iter().map(|value| value.powi(2)).sum::<f32>().sqrt() }

    // fn mae(&self, other: &SparseVector, n: usize) -> f32 { self.sad(other) / (n as f32) }
    fn sad(&self, other: &SparseVector) -> f32 {
        let (mut accumulator, mut refiter, mut cmpiter) = (0f32, 0usize, 0usize);

        while refiter < self.index.len() && cmpiter < other.index.len() {
            let (refidx, refval) = (self.index[refiter], self.value[refiter]);
            let (cmpidx, cmpval) = (other.index[cmpiter], other.value[cmpiter]);

            match refidx.cmp(&cmpidx) {
                Ordering::Equal => {
                    accumulator += (refval - cmpval).abs();
                    refiter += 1;
                    cmpiter += 1;
                },
                Ordering::Less => refiter += 1,
                Ordering::Greater => cmpiter += 1,
            };
        }

        accumulator
    }

    // fn mse(&self, other: &SparseVector, n: usize) -> f32 { self.ssd(other) / (n as f32) }
    fn euclidean(&self, other: &SparseVector) -> f32 { self.ssd(other).sqrt() }
    fn ssd(&self, other: &SparseVector) -> f32 {
        let (mut accumulator, mut refiter, mut cmpiter) = (0f32, 0usize, 0usize);

        while refiter < self.index.len() && cmpiter < other.index.len() {
            let (refidx, refval) = (self.index[refiter], self.value[refiter]);
            let (cmpidx, cmpval) = (other.index[cmpiter], other.value[cmpiter]);

            match refidx.cmp(&cmpidx) {
                Ordering::Equal => {
                    accumulator += (refval - cmpval).powi(2);
                    refiter += 1;
                    cmpiter += 1;
                },
                Ordering::Less => refiter += 1,
                Ordering::Greater => cmpiter += 1,
            };
        }

        accumulator
    }

    fn cosine(&self, other: &SparseVector) -> f32 {
        let (mut accumulator, mut refiter, mut cmpiter) = (0f32, 0usize, 0usize);

        while refiter < self.index.len() && cmpiter < other.index.len() {
            let (refidx, refval) = (self.index[refiter], self.value[refiter]);
            let (cmpidx, cmpval) = (other.index[cmpiter], other.value[cmpiter]);

            match refidx.cmp(&cmpidx) {
                Ordering::Equal => {
                    accumulator = refval.mul_add(cmpval, accumulator);
                    refiter += 1;
                    cmpiter += 1;
                },
                Ordering::Less => refiter += 1,
                Ordering::Greater => cmpiter += 1,
            };
        }

        accumulator
    }
}


impl Distances for DenseVector {
    fn norm(&self) -> f32 { self.value.iter().map(|value| value.powi(2)).sum::<f32>().sqrt() }

    fn sad(&self, other: &DenseVector) -> f32 {
        self.value.iter().zip(other.value.iter())
        .map(|(reference, comparison)| (reference - comparison).abs())
        .sum::<f32>()
    }

    fn euclidean(&self, other: &Self) -> f32{ self.ssd(other).sqrt() }
    fn ssd(&self, other: &DenseVector) -> f32 {
        self.value.iter().zip(other.value.iter())
        .map(|(reference, comparison)| (reference - comparison).powi(2))
        .sum::<f32>()
    }

    fn cosine(&self, other: &DenseVector) -> f32 {
        self.value.iter().zip(other.value.iter())
        .fold(0f32, |accumulator, (reference, comparison)| reference.mul_add(*comparison, accumulator))
    }
}
