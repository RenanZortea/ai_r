use rand::Rng;
use std::fmt;
use std::sync::Arc; // Requires: rand = "0.8" in Cargo.toml

// SENIOR PATTERN: Global Type Alias
// We defaults to f32 because LLMs are memory-bound.
// Using f64 doubles memory usage and halves SIMD throughput.
pub type Float = f32;

#[derive(Clone)]
pub struct Tensor {
    /// Underlying data. Wrapped in Arc for cheap cloning of the "view".
    /// This allows us to pass tensors around threads or functions without
    /// copying the massive weight buffers.
    data: Arc<Vec<Float>>,

    /// The dimensions of the tensor (e.g., [Batch, Seq, Hidden]).
    shape: Vec<usize>,

    /// The number of steps in memory to jump to the next element in each dimension.
    /// Essential for mapping N-D coordinates to 1-D memory.
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new Tensor from raw data and explicit shape.
    /// Panics if the data size does not match the product of the shape.
    pub fn new(data: Vec<Float>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            size,
            "Data length {} does not match shape {:?} (product: {})",
            data.len(),
            shape,
            size
        );

        let strides = Self::compute_strides(&shape);

        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    /// Creates a tensor of zeros with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![0.0 as Float; size], shape)
    }

    /// Creates a tensor with random values from a standard normal distribution.
    /// Essential for initializing weights if training from scratch.
    pub fn randn(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<Float> = (0..size)
            .map(|_| rng.gen_range(-1.0..1.0) as Float) // Simple initialization
            .collect();

        Self::new(data, shape)
    }

    /// Returns a reference to the underlying flat data slice.
    pub fn data(&self) -> &[Float] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the total number of elements.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Computes Row-Major strides (last dimension is contiguous).
    /// Shape: [2, 3] -> Strides: [3, 1]
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        if shape.is_empty() {
            return strides;
        }

        // Start from the second to last dimension
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

// Custom Debug implementation for cleaner terminal output
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, strides={:?}, device=CPU)",
            self.shape, self.strides
        )?;

        // Preview data if it's small enough
        if self.data.len() <= 10 {
            write!(f, "\nData: {:?}", self.data)
        } else {
            write!(f, "\nData: [First 5: {:?} ...]", &self.data[..5])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let t = Tensor::zeros(vec![2, 4]);
        assert_eq!(t.shape(), &[2, 4]);
        assert_eq!(t.size(), 8);
        assert_eq!(t.strides(), &[4, 1]);
    }

    #[test]
    fn test_randn() {
        let t = Tensor::randn(vec![10, 10]);
        assert_eq!(t.size(), 100);
        // Ensure not all zeros (probabilistic, but safe assumption)
        assert!(t.data()[0] != 0.0 || t.data()[1] != 0.0);
    }
}
