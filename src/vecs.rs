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
        if self.index.len() > other.index.len() {
            other.cosine(self)
        } else {
            // let mut reference = self.index.iter().zip(self.value.iter());
            // let mut comparison = other.index.iter().zip(other.value.iter());
            let mut accumulator = 0f32;
            let (mut i, mut j) = (0, 0);

            while i < self.index.len() && j < other.index.len() {
                let (a_index, a_value) = (self.index[i], self.value[i]);
                let (b_index, b_value) = (other.index[j], other.value[j]);

                match a_index.cmp(&b_index) {
                    Ordering::Equal => {
                        accumulator += a_value.mul_add(b_value, accumulator);
                        i += 1;
                        j += 1;
                    },
                    Ordering::Less => i += 1,
                    Ordering::Greater => j += 1,
                };
            }

            accumulator
        }
    }

    fn euclidean(&self, other: &SparseVector) -> f32 {
        if self.index.len() > other.index.len() {
            other.cosine(self)
        } else {
            self.index.iter()
            .enumerate()
            .map(|(reference, index)| {
                match other.index.binary_search(&index) {
                    Ok(comparison) => (self.value[reference] - other.value[comparison]).powi(2),
                    _ => 0f32,
                }
            })
            .sum::<f32>()
            .sqrt()
        }
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
