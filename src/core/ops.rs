use crate::core::tensor::{Float, Tensor};
use faer::Parallelism; // Removed unused MatRef/MatMut

impl Tensor {
    /// Perform Matrix Multiplication: C = A @ B
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // 1. Validation
        let a_shape = self.shape();
        let b_shape = other.shape();

        assert_eq!(a_shape.len(), 2, "A must be 2D");
        assert_eq!(b_shape.len(), 2, "B must be 2D");

        let m = a_shape[0];
        let k = a_shape[1];
        let k_b = b_shape[0];
        let n = b_shape[1];

        assert_eq!(
            k, k_b,
            "Dimension mismatch: A[{:?}] @ B[{:?}]",
            a_shape, b_shape
        );

        // 2. Create Output Tensor (C)
        let mut c = Tensor::zeros(vec![m, n]);

        // 3. The Math Kernel
        unsafe {
            // We use ::<Float> to explicitly tell faer we are working with f32.
            let a_view = faer::mat::from_raw_parts::<Float>(
                self.data().as_ptr(),
                m,
                k,
                self.strides()[0] as isize,
                self.strides()[1] as isize,
            );

            let b_view = faer::mat::from_raw_parts::<Float>(
                other.data().as_ptr(),
                k,
                n,
                other.strides()[0] as isize,
                other.strides()[1] as isize,
            );

            let c_ptr = c.data().as_ptr() as *mut Float;
            let c_view = faer::mat::from_raw_parts_mut::<Float>(
                c_ptr,
                m,
                n,
                c.strides()[0] as isize,
                c.strides()[1] as isize,
            );

            faer::linalg::matmul::matmul(
                c_view,
                a_view,
                b_view,
                None,
                1.0 as Float, // <--- CRITICAL FIX: explicit type match
                Parallelism::Rayon(0),
            );
        }

        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_simple() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = a.matmul(&b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_shapes() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]);
        let c = a.matmul(&b);

        assert_eq!(c.shape(), &[2, 1]);
        assert_eq!(c.data(), &[6.0, 15.0]);
    }
}
