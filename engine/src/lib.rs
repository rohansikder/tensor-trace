//! CPU-only inference for the original float32 llama2.c checkpoint format.
//! The model is immutable; every run owns its activations and KV cache.

mod model;
mod runtime;
mod sampling;
pub mod tokenizer;

pub use model::{Config, Model};
pub use runtime::Engine;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMetadata {
    pub name: String,
    pub architecture: String,
    pub config: Config,
    pub parameter_count: usize,
    pub model_bytes: usize,
    pub tokenizer_bytes: usize,
    pub engine: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub cache: bool,
    pub layer: usize,
    pub head: usize,
    pub trace: bool,
    pub stop_on_eos: bool,
}
impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: "Once upon a time".into(),
            max_tokens: 24,
            temperature: 0.0,
            top_p: 0.9,
            seed: 42,
            cache: true,
            layer: 0,
            head: 0,
            trace: true,
            stop_on_eos: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenInfo {
    pub id: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tokenization {
    pub ids: Vec<usize>,
    pub tokens: Vec<TokenInfo>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub id: usize,
    pub text: String,
    pub probability: f32,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StepTiming {
    pub forward_ms: f64,
    pub sampling_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StepTrace {
    /// Zero-based generated-token index. Position is the final input position.
    pub index: usize,
    pub position: usize,
    pub token: TokenInfo,
    pub text: String,
    pub context_tokens: Vec<TokenInfo>,
    /// Selected head's actual probabilities: query rows, key columns, zeros above diagonal.
    pub attention: Vec<Vec<f32>>,
    pub embedding: Vec<f32>,
    pub hidden_state: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_output: Vec<f32>,
    pub feed_forward: Vec<f32>,
    /// Unmodified-model softmax probabilities, before temperature/top-p sampling.
    pub top_tokens: Vec<Candidate>,
    pub entropy: f32,
    pub selected_probability: f32,
    pub layer: usize,
    pub head: usize,
    pub timing: StepTiming,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub forward_decode_ms: f64,
    pub time_to_first_token_ms: f64,
    pub sampling_ms: f64,
    pub total_ms: f64,
    pub tokens_per_second: f64,
    pub decode_ms_per_token: f64,
    pub kv_cache_bytes: usize,
    pub cache_enabled: bool,
    pub context_capacity: usize,
    pub stopped_reason: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationResult {
    pub text: String,
    pub generated_text: String,
    pub prompt_tokens: Vec<TokenInfo>,
    pub generated_tokens: Vec<TokenInfo>,
    pub steps: Vec<StepTrace>,
    pub stats: GenerationStats,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ForwardInspection {
    pub logits: Vec<f32>,
    pub attention: Vec<Vec<f32>>,
    pub embedding: Vec<f32>,
    pub hidden_state: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_output: Vec<f32>,
    pub feed_forward: Vec<f32>,
}
