//! Validated loading of legacy, float32 `llama2.c` checkpoints.
//!
//! Tensor ranges are checked before any weight allocation. The two historical
//! RoPE tables remain part of the file layout but are not learned parameters;
//! inference computes the rotations directly, as the reference runtime does.

use serde::Serialize;
use std::fmt;
use std::fs;
use std::path::Path;

const HEADER_BYTES: usize = 7 * 4;
const MAX_MODEL_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_CACHE_BYTES: usize = 512 * 1024 * 1024;
const MAX_VOCABULARY: usize = 1_000_000;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub shared_classifier: bool,
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.head_size() * self.n_kv_heads
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tensor {
    pub(crate) start: usize,
    pub(crate) len: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct Weights {
    pub(crate) token_embedding: Tensor,
    pub(crate) rms_att: Tensor,
    pub(crate) wq: Tensor,
    pub(crate) wk: Tensor,
    pub(crate) wv: Tensor,
    pub(crate) wo: Tensor,
    pub(crate) rms_ffn: Tensor,
    pub(crate) w1: Tensor,
    pub(crate) w2: Tensor,
    pub(crate) w3: Tensor,
    pub(crate) rms_final: Tensor,
    pub(crate) classifier: Tensor,
}

pub struct Model {
    pub(crate) config: Config,
    pub(crate) data: Vec<f32>,
    pub(crate) layout: Weights,
    pub(crate) model_bytes: usize,
    pub(crate) parameter_count: usize,
}

impl fmt::Debug for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Model")
            .field("config", &self.config)
            .field("model_bytes", &self.model_bytes)
            .field("parameter_count", &self.parameter_count)
            .finish_non_exhaustive()
    }
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)
            .map_err(|error| format!("Cannot read model {}: {error}", path.display()))?;
        if metadata.len() > MAX_MODEL_BYTES {
            return Err("Model file exceeds the 1 GiB safety limit".into());
        }
        let bytes = fs::read(path)
            .map_err(|error| format!("Cannot read model {}: {error}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() as u64 > MAX_MODEL_BYTES {
            return Err("Model file exceeds the 1 GiB safety limit".into());
        }
        if bytes.len() < HEADER_BYTES {
            return Err("Model checkpoint is truncated before its seven-integer header".into());
        }
        if bytes.starts_with(b"GGUF") || bytes.starts_with(b"GGML") {
            return Err(
                "Unsupported model format; expected a legacy llama2.c float32 checkpoint".into(),
            );
        }
        let mut integers = [0_i32; 7];
        for (index, value) in integers.iter_mut().enumerate() {
            *value = i32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap());
        }
        let positive = |index: usize, name: &str| -> Result<usize, String> {
            usize::try_from(integers[index])
                .ok()
                .filter(|&value| value > 0)
                .ok_or_else(|| format!("Model {name} must be positive"))
        };
        let vocab_size = integers[5]
            .checked_abs()
            .and_then(|size| usize::try_from(size).ok())
            .filter(|&size| (3..=MAX_VOCABULARY).contains(&size))
            .ok_or_else(|| "Model vocabulary size must be between 3 and 1,000,000".to_owned())?;
        let config = Config {
            dim: positive(0, "dimension")?,
            hidden_dim: positive(1, "hidden dimension")?,
            n_layers: positive(2, "layer count")?,
            n_heads: positive(3, "attention head count")?,
            n_kv_heads: positive(4, "KV head count")?,
            vocab_size,
            seq_len: positive(6, "context length")?,
            shared_classifier: integers[5] > 0,
        };
        if !config.dim.is_multiple_of(config.n_heads) {
            return Err("Model dimension must be divisible by its attention head count".into());
        }
        if !config.n_heads.is_multiple_of(config.n_kv_heads) {
            return Err("Attention head count must be divisible by its KV head count".into());
        }
        if !config.head_size().is_multiple_of(2) {
            return Err("Model attention head size must be even for RoPE rotations".into());
        }
        let cache_bytes = product(&[
            2,
            config.n_layers,
            config.seq_len,
            config.kv_dim(),
            std::mem::size_of::<f32>(),
        ])?;
        if cache_bytes > MAX_CACHE_BYTES {
            return Err("Model KV cache exceeds the 512 MiB safety limit".into());
        }
        let payload = &bytes[HEADER_BYTES..];
        if !payload.len().is_multiple_of(4) {
            return Err("Model payload has a partial float32 value or trailing bytes".into());
        }
        let count = payload.len() / 4;
        let mut ranges = Ranges { next: 0, count };
        let token_embedding = ranges.take(product(&[config.vocab_size, config.dim])?)?;
        let rms_att = ranges.take(product(&[config.n_layers, config.dim])?)?;
        let wq = ranges.take(product(&[config.n_layers, config.dim, config.dim])?)?;
        let wk = ranges.take(product(&[config.n_layers, config.kv_dim(), config.dim])?)?;
        let wv = ranges.take(product(&[config.n_layers, config.kv_dim(), config.dim])?)?;
        let wo = ranges.take(product(&[config.n_layers, config.dim, config.dim])?)?;
        let rms_ffn = ranges.take(product(&[config.n_layers, config.dim])?)?;
        let w1 = ranges.take(product(&[config.n_layers, config.hidden_dim, config.dim])?)?;
        let w2 = ranges.take(product(&[config.n_layers, config.dim, config.hidden_dim])?)?;
        let w3 = ranges.take(product(&[config.n_layers, config.hidden_dim, config.dim])?)?;
        let rms_final = ranges.take(config.dim)?;
        let rope_values = product(&[config.seq_len, config.head_size() / 2])?;
        ranges.take(rope_values)?;
        ranges.take(rope_values)?;
        let classifier = if config.shared_classifier {
            token_embedding
        } else {
            ranges.take(product(&[config.vocab_size, config.dim])?)?
        };
        if ranges.next != count {
            return Err(format!(
                "Model checkpoint has {} unexpected trailing float32 values",
                count - ranges.next
            ));
        }
        let parameter_count = count - 2 * rope_values;
        let mut data = Vec::with_capacity(count);
        for (index, raw) in payload.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes(raw.try_into().unwrap());
            if !value.is_finite() {
                return Err(format!("Model float32 value {index} is not finite"));
            }
            data.push(value);
        }
        Ok(Self {
            config,
            data,
            layout: Weights {
                token_embedding,
                rms_att,
                wq,
                wk,
                wv,
                wo,
                rms_ffn,
                w1,
                w2,
                w3,
                rms_final,
                classifier,
            },
            model_bytes: bytes.len(),
            parameter_count,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn model_bytes(&self) -> usize {
        self.model_bytes
    }

    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub(crate) fn tensor(&self, tensor: Tensor) -> &[f32] {
        &self.data[tensor.start..tensor.start + tensor.len]
    }
}

fn product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1_usize, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| "Model tensor dimensions overflow the supported size".to_owned())
    })
}

struct Ranges {
    next: usize,
    count: usize,
}

impl Ranges {
    fn take(&mut self, len: usize) -> Result<Tensor, String> {
        let end = self
            .next
            .checked_add(len)
            .filter(|&end| end <= self.count)
            .ok_or_else(|| {
                "Model checkpoint is truncated for its declared tensor dimensions".to_owned()
            })?;
        let tensor = Tensor {
            start: self.next,
            len,
        };
        self.next = end;
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(shared: bool) -> Vec<u8> {
        let header = [4_i32, 6, 2, 2, 1, if shared { 8 } else { -8 }, 3];
        let mut bytes = Vec::new();
        for value in header {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let count = if shared { 298 } else { 330 };
        for index in 0..count {
            bytes.extend_from_slice(&((index as f32 + 1.0) * 0.001).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn shared_classifier_aliases_embedding_and_excludes_rope_from_parameter_count() {
        let bytes = checkpoint(true);
        let model = Model::from_bytes(&bytes).unwrap();
        assert_eq!(model.config.head_size(), 2);
        assert_eq!(model.config.kv_dim(), 2);
        assert!(model.config.shared_classifier);
        assert_eq!(model.model_bytes(), bytes.len());
        assert_eq!(model.parameter_count(), 292);
        assert_eq!(model.layout.token_embedding, Tensor { start: 0, len: 32 });
        assert_eq!(model.layout.rms_att, Tensor { start: 32, len: 8 });
        assert_eq!(model.layout.rms_final, Tensor { start: 288, len: 4 });
        assert_eq!(model.layout.classifier, model.layout.token_embedding);
        assert_eq!(model.tensor(model.layout.classifier)[0], 0.001);
    }

    #[test]
    fn unshared_classifier_starts_after_both_reserved_rope_arrays() {
        let model = Model::from_bytes(&checkpoint(false)).unwrap();
        assert!(!model.config.shared_classifier);
        assert_eq!(model.parameter_count(), 324);
        assert_eq!(
            model.layout.classifier,
            Tensor {
                start: 298,
                len: 32
            }
        );
        assert!((model.tensor(model.layout.classifier)[0] - 0.299).abs() < 1e-6);
    }

    #[test]
    fn rejects_inconsistent_attention_dimensions() {
        for (index, value, message) in [
            (0, 5_i32, "dimension must be divisible"),
            (4, 3, "head count must be divisible"),
            (3, 4, "head size must be even"),
            (2, 0, "layer count must be positive"),
            (1, -1, "hidden dimension must be positive"),
            (5, i32::MIN, "vocabulary size"),
        ] {
            let mut bytes = checkpoint(true);
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            assert!(Model::from_bytes(&bytes).unwrap_err().contains(message));
        }
    }

    #[test]
    fn rejects_cache_allocations_above_the_bound_before_loading_weights() {
        let mut bytes = checkpoint(true);
        bytes[24..28].copy_from_slice(&(100_000_000_i32).to_le_bytes());
        assert!(Model::from_bytes(&bytes).unwrap_err().contains("KV cache"));
    }

    #[test]
    fn rejects_truncation_trailing_data_and_nonfinite_weights() {
        let good = checkpoint(true);
        for length in [0, HEADER_BYTES - 1, good.len() - 4] {
            assert!(Model::from_bytes(&good[..length]).is_err());
        }
        let mut extra_float = good.clone();
        extra_float.extend_from_slice(&0_f32.to_le_bytes());
        assert!(
            Model::from_bytes(&extra_float)
                .unwrap_err()
                .contains("trailing")
        );
        let mut extra_byte = good.clone();
        extra_byte.push(0);
        assert!(
            Model::from_bytes(&extra_byte)
                .unwrap_err()
                .contains("partial float32")
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = good.clone();
            invalid[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&value.to_le_bytes());
            assert!(
                Model::from_bytes(&invalid)
                    .unwrap_err()
                    .contains("not finite")
            );
        }
    }
}
