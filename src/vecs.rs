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
    fn cosine(&self, other: &Self) -> f32;  // assumes vectors are L2-normalised
    fn euclidean(&self, other: &Self) -> f32;
}


pub trait Transformations {
    fn normalize(&self) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
}


impl Distances for SparseVector {
    fn norm(&self) -> f32 { self.value.iter().map(|value| value.powi(2)).sum::<f32>().sqrt() }

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

    fn euclidean(&self, other: &SparseVector) -> f32 {
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

        accumulator.sqrt()
    }
}


impl Distances for DenseVector {
    fn norm(&self) -> f32 { self.value.iter().map(|value| value.powi(2)).sum::<f32>().sqrt() }

    fn cosine(&self, other: &DenseVector) -> f32 {
        self.value.iter().zip(other.value.iter())
        .fold(0f32, |accumulator, (reference, comparison)| reference.mul_add(*comparison, accumulator))
    }

    fn euclidean(&self, other: &DenseVector) -> f32 {
        self.value.iter().zip(other.value.iter())
        .map(|(reference, comparison)| (reference - comparison).powi(2))
        .sum::<f32>()
        .sqrt()
    }
}
