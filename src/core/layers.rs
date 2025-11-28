use crate::core::tensor::{Float, Tensor};

/// Trait that all Neural Network layers must implement
pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
}

pub struct Linear {
    pub weights: Tensor,
    // Bias is optional (Llama often skips bias in some layers)
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        // Weights initialized with normal distribution
        // Shape is [In, Out] for our simple matmul (PyTorch is usually [Out, In])
        // We stick to [In, Out] here for simplicity: X[B, In] @ W[In, Out] = Y[B, Out]
        Self {
            weights: Tensor::randn(vec![in_dim, out_dim]),
            bias: None,
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        // X @ W
        let mut out = x.matmul(&self.weights);

        // Add bias if it exists
        if let Some(b) = &self.bias {
            // This requires implementing add() or broadcast()
            // For now, we leave this as a placeholder or implement a simple loop
            // TODO: Implement broadcasting add
        }
        out
    }
}

/// RMSNorm: Root Mean Square Normalization
/// x_norm = x / sqrt(mean(x^2) + epsilon) * weight
pub struct RMSNorm {
    pub weight: Tensor, // The learnable gamma parameter
    pub eps: Float,
}

impl RMSNorm {
    pub fn new(dim: usize) -> Self {
        // Initialize weights to 1.0
        let data = vec![1.0; dim];
        Self {
            weight: Tensor::new(data, vec![dim]),
            eps: 1e-5,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_data = x.data();
        let w_data = self.weight.data();
        let shape = x.shape();

        // Assume last dimension is the feature dimension to normalize
        let hidden_dim = shape[shape.len() - 1];
        let num_tokens = x.size() / hidden_dim; // Batch * SeqLen

        let mut out_data = vec![0.0; x.size()];

        // Optimization: We iterate purely in simple loops for the compiler to vectorize
        for i in 0..num_tokens {
            let offset = i * hidden_dim;

            // 1. Calculate sum of squares
            let mut sum_sq = 0.0;
            for j in 0..hidden_dim {
                let val = x_data[offset + j];
                sum_sq += val * val;
            }

            // 2. Calculate RMS
            let rms = (sum_sq / hidden_dim as Float + self.eps).sqrt();
            let inv_rms = 1.0 / rms;

            // 3. Normalize and scale
            for j in 0..hidden_dim {
                out_data[offset + j] = x_data[offset + j] * inv_rms * w_data[j];
            }
        }

        Tensor::new(out_data, shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to compare two floats with tolerance
    fn assert_close(a: Float, b: Float, tol: Float) {
        if (a - b).abs() > tol {
            panic!(
                "Assertion failed: {} is not close to {} (tol {})",
                a, b, tol
            );
        }
    }

    #[test]
    fn test_linear_shape() {
        // Batch=2, InputFeatures=4
        let input = Tensor::randn(vec![2, 4]);

        // Linear: 4 -> 8
        let linear = Linear::new(4, 8);

        let out = linear.forward(&input);

        // Expected output: Batch=2, OutputFeatures=8
        assert_eq!(out.shape(), &[2, 8]);
    }

    #[test]
    fn test_rmsnorm_math() {
        // Create specific data: [2.0, 4.0]
        // Mean Square = (2^2 + 4^2) / 2 = (4 + 16) / 2 = 10
        // RMS = sqrt(10 + eps) ≈ 3.1622
        // InvRMS ≈ 0.31622
        // Out[0] = 2.0 * 0.31622 = 0.63245
        // Out[1] = 4.0 * 0.31622 = 1.26491

        let input = Tensor::new(vec![2.0, 4.0], vec![1, 2]);
        let rms = RMSNorm::new(2); // dim=2

        let out = rms.forward(&input);
        let data = out.data();

        // Check values with slight tolerance for float precision
        assert_close(data[0], 0.63245, 1e-4);
        assert_close(data[1], 1.26491, 1e-4);
    }

    #[test]
    fn test_rmsnorm_batching() {
        // Test if it handles multiple rows correctly
        // Row 1: [1.0, 1.0] -> RMS=1.0 -> Out=[1.0, 1.0]
        // Row 2: [2.0, 2.0] -> RMS=2.0 -> Out=[1.0, 1.0] (Normalization makes them unit scale)
        let input = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]);
        let rms = RMSNorm::new(2);

        let out = rms.forward(&input);
        let data = out.data();

        // All outputs should be roughly 1.0 (since they are equal values)
        for val in data.iter() {
            assert_close(*val, 1.0, 1e-3);
        }
    }
}
