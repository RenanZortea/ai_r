use burn::{
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    prelude::*,
};

// FIX: Added Default derive
#[derive(Config, Debug, Default)]
pub struct LlmConfig {
    #[config(default = 512)]
    pub d_model: usize,
    #[config(default = 10_000)]
    pub vocab_size: usize,
    #[config(default = 6)]
    pub n_layers: usize,
    #[config(default = 8)]
    pub n_heads: usize,
    #[config(default = 2048)]
    pub d_ff: usize,
    #[config(default = 128)]
    pub max_seq_len: usize,
}

#[derive(Module, Debug)]
pub struct Llm<B: Backend> {
    embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    output: Linear<B>,
}

impl LlmConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Llm<B> {
        Llm {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),

            transformer: TransformerEncoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .init(device),

            output: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend> Llm<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device(); // FIX: Capture device before moving input

        let x = self.embedding.forward(input);

        let x = self.transformer.forward(
            TransformerEncoderInput::new(x).mask_pad(self.generate_causal_mask(seq_len, &device)),
        );

        self.output.forward(x)
    }

    fn generate_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
        // FIX: Explicitly specify generic types <B, 2, Int> to avoid confusion
        Tensor::<B, 2, Int>::ones([seq_len, seq_len], device)
            .triu(1)
            .bool()
    }
}
