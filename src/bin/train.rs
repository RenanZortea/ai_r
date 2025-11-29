use ai_r::train;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};

fn main() {
    // 1. Define the Backend (WGPU for cross-platform GPU)
    // We wrap it in Autodiff to enable training (backpropagation)
    type MyBackend = Autodiff<Wgpu>;

    // 2. Select Device (Default picks the best available GPU)
    let device = WgpuDevice::default();

    println!("Starting training on device: {:?}", device);

    // 3. Run the library training function
    train::run::<MyBackend>(device);
}
