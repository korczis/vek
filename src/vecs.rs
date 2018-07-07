// index MUST be ordered and not contain repeated elements
// index.len() MUST be equal to value.len()
// it also assumes the vector has AT LEAST 1 dimension != 0
pub struct SparseVector {
    pub id: String,
    pub index: Vec<u32>,
    pub value: Vec<f32>,
}


// ALL vector comparisons must be between vectors of equal value.len() != 0
pub struct DenseVector {
    pub id: String,
    pub value: Vec<f32>,
}


trait Distances {
    fn norm(&self) -> f32;
    fn cosine(&self, other: &SparseVector) -> f32;
    fn euclidean(&self, other: &SparseVector) -> f32;
}


impl Distances for SparseVector {
    fn norm(&self) -> f32 { self.value.iter().map(|x| x.powi(2)).sum::<f32>().sqrt() }

    fn cosine(&self, other: &SparseVector) -> f32 {
        if self.index.len() > other.index.len() {
            other.cosine(self)
        } else {
            self.index.iter()
            .enumerate()
            .map(|(reference, index)| {
                match other.index.binary_search(&index) {
                    Ok(comparison) => self.value[reference] * other.value[comparison],
                    _ => 0f32,
                }
            })
            .sum()
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
                    Ok(comparison) => self.value[reference] - other.value[comparison],
                    _ => 0f32,
                }
            })
            .sum::<f32>()
            .sqrt()
        }
    }
}


impl Distances for DenseVector {
    fn norm(&self) -> f32 { self.value.iter().map(|x| x.powi(2)).sum::<f32>().sqrt() }

    fn cosine(&self, other: &SparseVector) -> f32 {
        self.value.iter().zip(other.value.iter())
        .map(|tup| tup.0 * tup.1)
        .sum()
    }

    fn euclidean(&self, other: &SparseVector) -> f32 {
        self.value.iter().zip(other.value.iter())
        .map(|tup| tup.0 - tup.1)
        .sum::<f32>()
        .sqrt()
    }
}
