use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct TextBatcher {
    tokenizer: Tokenizer,
    max_seq_len: usize,
}

impl TextBatcher {
    pub fn new(tokenizer: Tokenizer, max_seq_len: usize) -> Self {
        Self {
            tokenizer,
            max_seq_len,
        }
    }
}

/// The raw item coming from your dataset (just a string)
#[derive(Clone, Debug)]
pub struct TextItem {
    pub text: String,
}

/// The prepared batch ready for the GPU
#[derive(Clone, Debug)]
pub struct LlmBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,  // Inputs
    pub targets: Tensor<B, 2, Int>, // Next-token targets
}

// FIX: Added 'B' as the first generic argument to Batcher
impl<B: Backend> Batcher<B, TextItem, LlmBatch<B>> for TextBatcher {
    fn batch(&self, items: Vec<TextItem>, device: &B::Device) -> LlmBatch<B> {
        // 1. Extract strings
        let texts: Vec<String> = items.into_iter().map(|item| item.text).collect();

        // 2. Encode using HuggingFace Tokenizer
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .expect("Tokenization failed");

        // 3. Collect Token IDs into a flattened vector
        let mut all_ids: Vec<i32> = Vec::new();
        for encoding in encodings {
            let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
            let len = ids.len().min(self.max_seq_len + 1);
            all_ids.extend_from_slice(&ids[..len]);
        }

        // 4. Create Tensor from flattened data
        let batch_size = all_ids.len() / (self.max_seq_len + 1);

        // Safety check to prevent panic on empty batch
        if batch_size == 0 {
            panic!("Batch size calculated to 0. Check max_seq_len vs input text length.");
        }

        let tensor = Tensor::<B, 1, Int>::from_ints(all_ids.as_slice(), device)
            .reshape([batch_size, self.max_seq_len + 1]);

        // 5. Shift for Causal Modeling
        let tokens = tensor.clone().slice([0..batch_size, 0..self.max_seq_len]);
        let targets = tensor.slice([0..batch_size, 1..self.max_seq_len + 1]);

        LlmBatch { tokens, targets }
    }
}
