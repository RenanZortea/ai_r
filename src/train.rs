use crate::{
    data::LlmBatch,
    model::{Llm, LlmConfig},
};
use burn::{
    optim::AdamWConfig,
    prelude::*,
    tensor::backend::AutodiffBackend,
    // FIX: Switched to ClassificationOutput (Correct for LLMs)
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

// Define what happens during a training step
impl<B: AutodiffBackend> TrainStep<LlmBatch<B>, ClassificationOutput<B>> for Llm<B> {
    fn step(&self, batch: LlmBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let logits = self.forward(batch.tokens);

        let [batch_size, seq_len, vocab_size] = logits.dims();

        // FIX: Capture device before 'logits' is moved into 'reshape'
        let device = logits.device();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits_flat.clone(), targets_flat.clone());

        // FIX: Calculate gradients! (This was missing)
        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            ClassificationOutput::new(loss, logits_flat, targets_flat),
        )
    }
}

// Define validation step
impl<B: Backend> ValidStep<LlmBatch<B>, ClassificationOutput<B>> for Llm<B> {
    fn step(&self, batch: LlmBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.tokens);

        let [batch_size, seq_len, vocab_size] = logits.dims();

        // FIX: Capture device before 'logits' is moved
        let device = logits.device();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let config = LlmConfig {
        d_model: 256,
        n_layers: 4,
        vocab_size: 1000,
        ..Default::default()
    };

    let _model = config.init::<B>(&device);

    // FIX: Added type annotations <B, Llm<B>> so compiler knows what we are optimizing
    let _optimizer = AdamWConfig::new().init::<B, Llm<B>>();

    println!("Model initialized on RTX 2060 (WGPU): {:?}", config);
}
