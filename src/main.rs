use ai_r::core::tensor::Tensor;

fn main() {
    println!("--- AI_R Engine Initialized ---");

    // Create two random tensors
    let a = Tensor::randn(vec![128, 256]);
    let b = Tensor::randn(vec![256, 128]);

    println!("Matrix A: {:?}", a);
    println!("Matrix B: {:?}", b);

    // Run the engine
    let c = a.matmul(&b);

    println!("Result C (Shape {:?}):", c.shape());
    // We print the debug view, which shows the first few elements
    println!("{:?}", c);
}
