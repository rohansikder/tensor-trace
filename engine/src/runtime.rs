use crate::model::Model;
use crate::sampling::{Sampler, softmax};
use crate::tokenizer::Tokenizer;
use crate::{
    Candidate, ForwardInspection, GenerationRequest, GenerationResult, GenerationStats,
    ModelMetadata, StepTiming, StepTrace, TokenInfo, Tokenization,
};
use std::path::Path;
use std::time::Instant;

const MAX_PROMPT_BYTES: usize = 8192;
const MAX_TRACE_CONTEXT: usize = 512;
const MAX_TRACE_CELLS: usize = 4_194_304;

pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    name: String,
    tokenizer_bytes: usize,
}

impl Engine {
    pub fn load(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let model = Model::load(model_path.as_ref())?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref(), model.config.vocab_size)?;
        let tokenizer_bytes = std::fs::metadata(tokenizer_path.as_ref())
            .map_err(|e| format!("Cannot read tokenizer metadata: {e}"))?
            .len() as usize;
        let name = model_path
            .as_ref()
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            model,
            tokenizer,
            name,
            tokenizer_bytes,
        })
    }
    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            name: self.name.clone(),
            architecture: "llama".into(),
            config: self.model.config.clone(),
            parameter_count: self.model.parameter_count,
            model_bytes: self.model.model_bytes,
            tokenizer_bytes: self.tokenizer_bytes,
            engine: "Rust CPU · float32".into(),
        }
    }
    pub fn tokenize(&self, text: &str) -> Result<Tokenization, String> {
        if text.len() > MAX_PROMPT_BYTES {
            return Err("Prompt exceeds the 8192-byte inference limit".into());
        }
        let ids = self.tokenizer.encode(text, true, false)?;
        Ok(Tokenization {
            tokens: self.token_infos(&ids),
            ids,
        })
    }
    fn token_infos(&self, ids: &[usize]) -> Vec<TokenInfo> {
        ids.iter()
            .map(|&id| TokenInfo {
                id,
                text: self.tokenizer.token_text(id),
            })
            .collect()
    }
    pub fn inspect_tokens(
        &self,
        ids: &[usize],
        cache: bool,
        layer: usize,
        head: usize,
    ) -> Result<ForwardInspection, String> {
        self.validate_tokens(ids, layer, head)?;
        if ids.len() > MAX_TRACE_CONTEXT {
            return Err("Attention inspection is limited to 512 context tokens".into());
        }
        let mut session = Session::new(&self.model, layer, head, true, ids.len());
        if cache {
            for (position, &id) in ids.iter().enumerate() {
                session.forward(id, position)?;
            }
        } else {
            for end in 0..ids.len() {
                for (position, &id) in ids[..=end].iter().enumerate() {
                    session.forward(id, position)?;
                }
            }
        }
        Ok(session.inspection(ids.len()))
    }
    fn validate_tokens(&self, ids: &[usize], layer: usize, head: usize) -> Result<(), String> {
        let c = &self.model.config;
        if ids.is_empty() {
            return Err("Input must contain at least one token".into());
        }
        if ids.len() > c.seq_len {
            return Err(format!(
                "Input exceeds the {} token context capacity",
                c.seq_len
            ));
        }
        if ids.iter().any(|&id| id >= c.vocab_size) {
            return Err("Input contains an out-of-vocabulary token ID".into());
        }
        if layer >= c.n_layers || head >= c.n_heads {
            return Err("Selected layer or head is outside the model configuration".into());
        }
        Ok(())
    }
    pub fn generate(&self, request: &GenerationRequest) -> Result<GenerationResult, String> {
        self.generate_stream(request, |_| true)
    }
    pub fn generate_tokens(
        &self,
        ids: &[usize],
        request: &GenerationRequest,
    ) -> Result<GenerationResult, String> {
        self.generate_impl(ids, request, |_| true)
    }
    pub fn generate_tokens_stream(
        &self,
        ids: &[usize],
        request: &GenerationRequest,
        callback: impl FnMut(&StepTrace) -> bool,
    ) -> Result<GenerationResult, String> {
        self.generate_impl(ids, request, callback)
    }
    pub fn generate_stream(
        &self,
        request: &GenerationRequest,
        callback: impl FnMut(&StepTrace) -> bool,
    ) -> Result<GenerationResult, String> {
        if request.prompt.len() > MAX_PROMPT_BYTES {
            return Err("Prompt exceeds the 8192-byte inference limit".into());
        }
        let ids = self.tokenizer.encode(&request.prompt, true, false)?;
        self.generate_impl(&ids, request, callback)
    }
    fn generate_impl(
        &self,
        input_ids: &[usize],
        request: &GenerationRequest,
        mut callback: impl FnMut(&StepTrace) -> bool,
    ) -> Result<GenerationResult, String> {
        self.validate_tokens(input_ids, request.layer, request.head)?;
        if request.max_tokens == 0 || request.max_tokens > 512 {
            return Err("maxTokens must be between 1 and 512".into());
        }
        if input_ids.len() >= self.model.config.seq_len {
            return Err("Prompt leaves no context space for generation".into());
        }
        // Validate sampler settings even if generation would immediately stop on EOS.
        let mut sampler = Sampler::new(request.seed);
        sampler.sample(&[0.0], request.temperature, request.top_p)?;
        sampler = Sampler::new(request.seed);
        let budget = request
            .max_tokens
            .min(self.model.config.seq_len - input_ids.len());
        let trace_capacity = input_ids.len() + budget;
        if request.trace
            && (trace_capacity > MAX_TRACE_CONTEXT
                || trace_capacity
                    .checked_mul(trace_capacity)
                    .and_then(|n| n.checked_mul(budget))
                    .is_none_or(|n| n > MAX_TRACE_CELLS))
        {
            return Err("Trace exceeds the bounded context/snapshot budget; shorten the prompt or generation, or disable tracing".into());
        }
        let mut session = Session::new(
            &self.model,
            request.layer,
            request.head,
            request.trace,
            input_ids.len() + budget,
        );
        let mut ids = input_ids.to_vec();
        let mut generated = Vec::with_capacity(budget);
        let mut steps = Vec::with_capacity(budget);
        let mut generated_bytes = Vec::new();
        let mut previous = *ids.last().unwrap();
        let mut prefill_ms = 0.0;
        let mut decode_ms = 0.0;
        let mut forward_decode_ms = 0.0;
        let mut sampling_ms = 0.0;
        let mut reason = if budget < request.max_tokens {
            "context_limit"
        } else {
            "max_tokens"
        }
        .to_owned();
        let started = Instant::now();
        for index in 0..budget {
            let forward_started = Instant::now();
            if !request.cache || index == 0 {
                // Deliberately recompute every input position on uncached runs. No
                // stored activations or previous-step K/V participate in this pass.
                for (position, &id) in ids.iter().enumerate() {
                    session.forward(id, position)?;
                }
            } else {
                session.forward(*ids.last().unwrap(), ids.len() - 1)?;
            }
            let forward_ms = forward_started.elapsed().as_secs_f64() * 1000.0;
            if index == 0 {
                prefill_ms = forward_ms;
            } else {
                decode_ms += forward_ms;
                forward_decode_ms += forward_ms;
            }
            let sample_started = Instant::now();
            let next = sampler.sample(&session.logits, request.temperature, request.top_p)?;
            let mut probabilities = session.logits.clone();
            softmax(&mut probabilities)?;
            let entropy = -probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f32>();
            let mut order: Vec<usize> = (0..probabilities.len()).collect();
            let top_count = 8.min(order.len());
            order.select_nth_unstable_by(top_count - 1, |&a, &b| {
                probabilities[b]
                    .total_cmp(&probabilities[a])
                    .then(a.cmp(&b))
            });
            order[..top_count].sort_unstable_by(|&a, &b| {
                probabilities[b]
                    .total_cmp(&probabilities[a])
                    .then(a.cmp(&b))
            });
            let top_tokens = order[..top_count]
                .iter()
                .map(|&id| Candidate {
                    id,
                    text: self.tokenizer.token_text(id),
                    probability: probabilities[id],
                    logit: session.logits[id],
                })
                .collect();
            let sampling_time = sample_started.elapsed().as_secs_f64() * 1000.0;
            sampling_ms += sampling_time;
            if index > 0 {
                decode_ms += sampling_time;
            }
            let token = TokenInfo {
                id: next,
                text: self.tokenizer.token_text(next),
            };
            generated.push(token.clone());
            if next != 1 && next != 2 {
                generated_bytes.extend(self.tokenizer.decode_piece(previous, next)?);
            }
            let snapshot = if request.trace {
                session.inspection(ids.len())
            } else {
                ForwardInspection::empty()
            };
            let step = StepTrace {
                index,
                position: ids.len() - 1,
                token,
                text: String::from_utf8_lossy(&generated_bytes).into_owned(),
                context_tokens: if request.trace {
                    self.token_infos(&ids)
                } else {
                    vec![]
                },
                attention: snapshot.attention,
                embedding: snapshot.embedding,
                hidden_state: snapshot.hidden_state,
                query: snapshot.query,
                key: snapshot.key,
                value: snapshot.value,
                attention_output: snapshot.attention_output,
                feed_forward: snapshot.feed_forward,
                top_tokens,
                entropy,
                selected_probability: probabilities[next],
                layer: request.layer,
                head: request.head,
                timing: StepTiming {
                    forward_ms,
                    sampling_ms: sampling_time,
                    total_ms: forward_ms + sampling_time,
                },
            };
            let keep_going = callback(&step);
            steps.push(step);
            previous = next;
            if !keep_going {
                reason = "cancelled".into();
                break;
            }
            if next == 2 && request.stop_on_eos {
                reason = "eos".into();
                break;
            }
            ids.push(next);
        }
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        let generated_text = String::from_utf8_lossy(&generated_bytes).into_owned();
        let prompt_text = self.tokenizer.decode(input_ids)?;
        let decode_tokens = generated.len().saturating_sub(1);
        // Active K/V payload, not allocator/RSS memory. Uncached evaluation also
        // needs transient K/V scratch, but retains no reusable cache between steps.
        let active_positions = input_ids.len() + generated.len().saturating_sub(1);
        let kv_cache_bytes = if request.cache {
            2 * self.model.config.n_layers
                * active_positions
                * self.model.config.kv_dim()
                * std::mem::size_of::<f32>()
        } else {
            0
        };
        let stats = GenerationStats {
            prompt_tokens: input_ids.len(),
            generated_tokens: generated.len(),
            decode_tokens,
            prefill_ms,
            decode_ms,
            forward_decode_ms,
            time_to_first_token_ms: steps.first().map_or(0.0, |step| step.timing.total_ms),
            sampling_ms,
            total_ms,
            tokens_per_second: if decode_tokens > 0 && decode_ms > 0.0 {
                decode_tokens as f64 * 1000.0 / decode_ms
            } else {
                0.0
            },
            decode_ms_per_token: if decode_tokens > 0 {
                decode_ms / decode_tokens as f64
            } else {
                0.0
            },
            kv_cache_bytes,
            cache_enabled: request.cache,
            context_capacity: self.model.config.seq_len,
            stopped_reason: reason,
            seed: request.seed,
        };
        Ok(GenerationResult {
            text: format!("{prompt_text}{generated_text}"),
            generated_text,
            prompt_tokens: self.token_infos(input_ids),
            generated_tokens: generated,
            steps,
            stats,
        })
    }
}

impl ForwardInspection {
    fn empty() -> Self {
        Self {
            logits: vec![],
            attention: vec![],
            embedding: vec![],
            hidden_state: vec![],
            query: vec![],
            key: vec![],
            value: vec![],
            attention_output: vec![],
            feed_forward: vec![],
        }
    }
}

fn rmsnorm(out: &mut [f32], input: &[f32], weight: &[f32]) {
    let ss = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (ss + 1e-5).sqrt();
    for ((out, &input), &weight) in out.iter_mut().zip(input).zip(weight) {
        *out = input * scale * weight;
    }
}
fn matmul(out: &mut [f32], input: &[f32], weight: &[f32]) {
    for (row, out) in weight.chunks_exact(input.len()).zip(out) {
        // Four independent accumulators permit CPU auto-vectorization without
        // architecture-specific intrinsics or relaxed/unsafe floating-point flags.
        let mut sums = [0.0f32; 4];
        let chunks = input.len() / 4;
        for i in 0..chunks {
            let p = i * 4;
            sums[0] += row[p] * input[p];
            sums[1] += row[p + 1] * input[p + 1];
            sums[2] += row[p + 2] * input[p + 2];
            sums[3] += row[p + 3] * input[p + 3];
        }
        let mut sum = sums.iter().sum::<f32>();
        for i in chunks * 4..input.len() {
            sum += row[i] * input[i];
        }
        *out = sum;
    }
}

struct Session<'a> {
    model: &'a Model,
    layer: usize,
    head: usize,
    capture: bool,
    capacity: usize,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    logits: Vec<f32>,
    matrix: Vec<f32>,
    snapshot: ForwardInspection,
}
impl<'a> Session<'a> {
    fn new(model: &'a Model, layer: usize, head: usize, capture: bool, capacity: usize) -> Self {
        let c = &model.config;
        let kv_len = c.n_layers * capacity * c.kv_dim();
        Self {
            model,
            layer,
            head,
            capture,
            capacity,
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            q: vec![0.0; c.dim],
            k: vec![0.0; c.kv_dim()],
            v: vec![0.0; c.kv_dim()],
            att: vec![0.0; c.n_heads * capacity],
            key_cache: vec![0.0; kv_len],
            value_cache: vec![0.0; kv_len],
            logits: vec![0.0; c.vocab_size],
            matrix: if capture {
                vec![0.0; capacity * capacity]
            } else {
                vec![]
            },
            snapshot: ForwardInspection::empty(),
        }
    }
    fn forward(&mut self, token: usize, pos: usize) -> Result<(), String> {
        let m = self.model;
        let c = &m.config;
        let d = c.dim;
        let kv = c.kv_dim();
        let hs = c.head_size();
        let kv_mul = c.n_heads / c.n_kv_heads;
        if token >= c.vocab_size || pos >= self.capacity {
            return Err("Forward input outside model bounds".into());
        }
        self.x
            .copy_from_slice(&m.tensor(m.layout.token_embedding)[token * d..(token + 1) * d]);
        if self.capture {
            self.snapshot.embedding = self.x.iter().take(24).copied().collect();
        }
        for layer in 0..c.n_layers {
            rmsnorm(
                &mut self.xb,
                &self.x,
                &m.tensor(m.layout.rms_att)[layer * d..(layer + 1) * d],
            );
            matmul(
                &mut self.q,
                &self.xb,
                &m.tensor(m.layout.wq)[layer * d * d..(layer + 1) * d * d],
            );
            matmul(
                &mut self.k,
                &self.xb,
                &m.tensor(m.layout.wk)[layer * kv * d..(layer + 1) * kv * d],
            );
            matmul(
                &mut self.v,
                &self.xb,
                &m.tensor(m.layout.wv)[layer * kv * d..(layer + 1) * kv * d],
            );
            for i in (0..d).step_by(2) {
                let freq = 1.0 / 10000.0f32.powf((i % hs) as f32 / hs as f32);
                let val = pos as f32 * freq;
                let (sin, cos) = val.sin_cos();
                let q0 = self.q[i];
                let q1 = self.q[i + 1];
                self.q[i] = q0 * cos - q1 * sin;
                self.q[i + 1] = q0 * sin + q1 * cos;
                if i < kv {
                    let k0 = self.k[i];
                    let k1 = self.k[i + 1];
                    self.k[i] = k0 * cos - k1 * sin;
                    self.k[i + 1] = k0 * sin + k1 * cos;
                }
            }
            let offset = layer * self.capacity * kv + pos * kv;
            self.key_cache[offset..offset + kv].copy_from_slice(&self.k);
            self.value_cache[offset..offset + kv].copy_from_slice(&self.v);
            if self.capture && layer == self.layer {
                let start = self.head * hs;
                let ks = (self.head / kv_mul) * hs;
                self.snapshot.query = self.q[start..start + hs].iter().take(24).copied().collect();
                self.snapshot.key = self.k[ks..ks + hs].iter().take(24).copied().collect();
                self.snapshot.value = self.v[ks..ks + hs].iter().take(24).copied().collect();
            }
            for head in 0..c.n_heads {
                let qs = head * hs;
                let ks = (head / kv_mul) * hs;
                let att = &mut self.att[head * self.capacity..head * self.capacity + pos + 1];
                for (t, score) in att.iter_mut().enumerate() {
                    let key = layer * self.capacity * kv + t * kv + ks;
                    let mut sum = 0.0;
                    for i in 0..hs {
                        sum += self.q[qs + i] * self.key_cache[key + i];
                    }
                    *score = sum / (hs as f32).sqrt();
                }
                softmax(att)?;
                self.xb[qs..qs + hs].fill(0.0);
                for (t, &probability) in att.iter().enumerate() {
                    let value = layer * self.capacity * kv + t * kv + ks;
                    for i in 0..hs {
                        self.xb[qs + i] += probability * self.value_cache[value + i];
                    }
                }
                if self.capture && layer == self.layer && head == self.head {
                    self.matrix[pos * self.capacity..pos * self.capacity + pos + 1]
                        .copy_from_slice(att);
                }
            }
            matmul(
                &mut self.xb2,
                &self.xb,
                &m.tensor(m.layout.wo)[layer * d * d..(layer + 1) * d * d],
            );
            if self.capture && layer == self.layer {
                self.snapshot.attention_output = self.xb2.iter().take(24).copied().collect();
            }
            for i in 0..d {
                self.x[i] += self.xb2[i];
            }
            rmsnorm(
                &mut self.xb,
                &self.x,
                &m.tensor(m.layout.rms_ffn)[layer * d..(layer + 1) * d],
            );
            let hd = c.hidden_dim;
            matmul(
                &mut self.hb,
                &self.xb,
                &m.tensor(m.layout.w1)[layer * hd * d..(layer + 1) * hd * d],
            );
            matmul(
                &mut self.hb2,
                &self.xb,
                &m.tensor(m.layout.w3)[layer * hd * d..(layer + 1) * hd * d],
            );
            for i in 0..hd {
                let value = self.hb[i];
                self.hb[i] = value / (1.0 + (-value).exp()) * self.hb2[i];
            }
            matmul(
                &mut self.xb,
                &self.hb,
                &m.tensor(m.layout.w2)[layer * d * hd..(layer + 1) * d * hd],
            );
            if self.capture && layer == self.layer {
                self.snapshot.feed_forward = self.xb.iter().take(24).copied().collect();
            }
            for i in 0..d {
                self.x[i] += self.xb[i];
            }
        }
        let input = self.x.clone();
        rmsnorm(&mut self.x, &input, m.tensor(m.layout.rms_final));
        matmul(&mut self.logits, &self.x, m.tensor(m.layout.classifier));
        if self.logits.iter().any(|v| !v.is_finite()) {
            return Err("Model produced non-finite logits".into());
        }
        if self.capture {
            self.snapshot.hidden_state = self.x.iter().take(24).copied().collect();
        }
        Ok(())
    }
    fn inspection(&self, len: usize) -> ForwardInspection {
        ForwardInspection {
            logits: self.logits.clone(),
            attention: if self.capture {
                (0..len)
                    .map(|row| self.matrix[row * self.capacity..row * self.capacity + len].to_vec())
                    .collect()
            } else {
                vec![]
            },
            embedding: self.snapshot.embedding.clone(),
            hidden_state: self.snapshot.hidden_state.clone(),
            query: self.snapshot.query.clone(),
            key: self.snapshot.key.clone(),
            value: self.snapshot.value.clone(),
            attention_output: self.snapshot.attention_output.clone(),
            feed_forward: self.snapshot.feed_forward.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_multiplication_supports_odd_inner_dimensions() {
        let mut out = [0.0; 2];
        matmul(&mut out, &[1.0, 2.0, 3.0], &[1.0, 0.0, 1.0, 2.0, 1.0, 0.0]);
        assert_eq!(out, [4.0, 4.0]);
    }
    #[test]
    fn rms_norm_has_unit_mean_square_before_weighting() {
        let mut out = [0.0; 4];
        rmsnorm(&mut out, &[1.0, 2.0, 3.0, 4.0], &[1.0; 4]);
        assert!((out.iter().map(|v| v * v).sum::<f32>() / 4.0 - 1.0).abs() < 1e-5);
    }
}

#[cfg(test)]
mod behavior_tests {
    use super::*;
    const SHARED: &[u8] = include_bytes!("../../validation/fixtures/tiny-shared.bin");
    const UNSHARED: &[u8] = include_bytes!("../../validation/fixtures/tiny-unshared.bin");
    const VOCAB: &[u8] = include_bytes!("../../validation/fixtures/tokenizer.bin");

    fn engine() -> Engine {
        Engine {
            model: Model::from_bytes(SHARED).unwrap(),
            tokenizer: Tokenizer::from_bytes(VOCAB, 512).unwrap(),
            name: "deterministic test fixture".into(),
            tokenizer_bytes: VOCAB.len(),
        }
    }
    fn request() -> GenerationRequest {
        GenerationRequest {
            prompt: "fixture".into(),
            max_tokens: 8,
            stop_on_eos: false,
            layer: 1,
            head: 3,
            ..GenerationRequest::default()
        }
    }
    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn cached_and_recomputed_prefixes_have_identical_logits_and_attention() {
        let engine = engine();
        let ids = [1, 42, 17, 93, 80];
        for layer in 0..2 {
            for head in 0..4 {
                let a = engine.inspect_tokens(&ids, true, layer, head).unwrap();
                let b = engine.inspect_tokens(&ids, false, layer, head).unwrap();
                close(&a.logits, &b.logits);
                for (a, b) in a.attention.iter().zip(&b.attention) {
                    close(a, b);
                }
            }
        }
    }
    #[test]
    fn attention_is_normalized_causal_and_earlier_rows_ignore_future_tokens() {
        let engine = engine();
        let a = engine.inspect_tokens(&[1, 42, 17, 93], true, 1, 2).unwrap();
        let b = engine
            .inspect_tokens(&[1, 42, 17, 100], true, 1, 2)
            .unwrap();
        for (row, values) in a.attention.iter().enumerate() {
            assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!(values[row + 1..].iter().all(|&v| v == 0.0));
            if row < 3 {
                close(values, &b.attention[row]);
            }
        }
    }
    #[test]
    fn grouped_query_heads_share_keys_and_values_but_have_distinct_queries() {
        let engine = engine();
        let a = engine.inspect_tokens(&[1, 42, 17], true, 0, 0).unwrap();
        let b = engine.inspect_tokens(&[1, 42, 17], true, 0, 1).unwrap();
        close(&a.key, &b.key);
        close(&a.value, &b.value);
        assert!(
            a.query
                .iter()
                .zip(&b.query)
                .any(|(a, b)| (a - b).abs() > 1e-5)
        );
    }
    #[test]
    fn complete_generation_matches_with_and_without_kv_reuse() {
        let engine = engine();
        let ids = [1, 42, 17];
        let on = engine.generate_tokens(&ids, &request()).unwrap();
        let mut off_request = request();
        off_request.cache = false;
        let off = engine.generate_tokens(&ids, &off_request).unwrap();
        assert_eq!(
            on.generated_tokens.iter().map(|t| t.id).collect::<Vec<_>>(),
            off.generated_tokens
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>()
        );
        for (a, b) in on.steps.iter().zip(&off.steps) {
            assert!((a.selected_probability - b.selected_probability).abs() < 1e-7);
            close(&a.hidden_state, &b.hidden_state);
        }
        assert!(on.stats.kv_cache_bytes > 0);
        assert_eq!(off.stats.kv_cache_bytes, 0);
    }
    #[test]
    fn seeded_generation_owns_rng_state_and_context_limits_are_reported() {
        let engine = engine();
        let mut req = request();
        req.temperature = 0.8;
        let a = engine.generate_tokens(&[1, 42], &req).unwrap();
        let b = engine.generate_tokens(&[1, 42], &req).unwrap();
        assert_eq!(
            a.generated_tokens.iter().map(|t| t.id).collect::<Vec<_>>(),
            b.generated_tokens.iter().map(|t| t.id).collect::<Vec<_>>()
        );
        req.max_tokens = 40;
        let result = engine.generate_tokens(&[1, 42], &req).unwrap();
        assert_eq!(result.generated_tokens.len(), 30);
        assert_eq!(result.stats.stopped_reason, "context_limit");
    }
    #[test]
    fn cancellation_callbacks_run_without_tracing_and_stop_the_run() {
        let engine = engine();
        let mut req = request();
        req.trace = false;
        let mut calls = 0;
        let result = engine
            .generate_tokens_stream(&[1, 42], &req, |step| {
                calls += 1;
                assert!(step.attention.is_empty());
                calls < 3
            })
            .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(result.generated_tokens.len(), 3);
        assert_eq!(result.stats.stopped_reason, "cancelled");
    }
    #[test]
    fn timing_metrics_include_sampling_and_keep_prefill_separate() {
        let result = engine().generate_tokens(&[1, 42], &request()).unwrap();
        let steady = result
            .steps
            .iter()
            .skip(1)
            .map(|s| s.timing.total_ms)
            .sum::<f64>();
        assert!((result.stats.decode_ms - steady).abs() < 1e-6);
        assert!(result.stats.decode_ms >= result.stats.forward_decode_ms);
        assert_eq!(result.stats.decode_tokens, 7);
        assert!(result.stats.time_to_first_token_ms >= result.stats.prefill_ms);
        assert!(
            (result.stats.tokens_per_second * result.stats.decode_ms / 1000.0 - 7.0).abs() < 1e-6
        );
        assert_eq!(result.stats.kv_cache_bytes, 2 * 2 * (2 + 8 - 1) * 8 * 4);
    }
    #[test]
    fn eos_stops_normal_generation_but_not_a_fixed_budget_benchmark() {
        let mut model = Model::from_bytes(UNSHARED).unwrap();
        let hidden = {
            let mut session = Session::new(&model, 0, 0, true, 4);
            session.forward(1, 0).unwrap();
            session.forward(42, 1).unwrap();
            session.inspection(2).hidden_state
        };
        let start = model.layout.classifier.start;
        let len = model.layout.classifier.len;
        model.data[start..start + len].fill(0.0);
        let d = model.config.dim;
        model.data[start + 2 * d..start + 3 * d].copy_from_slice(&hidden);
        let engine = Engine {
            model,
            tokenizer: Tokenizer::from_bytes(VOCAB, 512).unwrap(),
            name: "EOS fixture".into(),
            tokenizer_bytes: VOCAB.len(),
        };
        let mut req = request();
        req.stop_on_eos = true;
        let result = engine.generate_tokens(&[1, 42], &req).unwrap();
        assert_eq!(result.generated_tokens.len(), 1);
        assert_eq!(result.generated_tokens[0].id, 2);
        assert_eq!(result.stats.stopped_reason, "eos");
        req.stop_on_eos = false;
        assert_eq!(
            engine
                .generate_tokens(&[1, 42], &req)
                .unwrap()
                .generated_tokens
                .len(),
            8
        );
    }
    #[test]
    fn public_boundaries_reject_bad_ids_heads_and_generation_options() {
        let engine = engine();
        assert!(engine.inspect_tokens(&[], true, 0, 0).is_err());
        assert!(engine.inspect_tokens(&[512], true, 0, 0).is_err());
        assert!(engine.inspect_tokens(&[1], true, 2, 0).is_err());
        assert!(engine.inspect_tokens(&[1], true, 0, 4).is_err());
        let mut req = request();
        req.max_tokens = 0;
        assert!(engine.generate_tokens(&[1], &req).is_err());
        req.max_tokens = 513;
        assert!(engine.generate_tokens(&[1], &req).is_err());
        assert!(engine.tokenize(&"a".repeat(MAX_PROMPT_BYTES + 1)).is_err());
    }
}
