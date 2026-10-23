# TensorTrace implementation plan

## Goal

A CPU language-model runtime in Rust with a guided visual explanation of its actual forward pass. A separate comparison view demonstrates the memory and computation tradeoff of KV caching.

## Delivery

1. Load a pinned, small pretrained Llama-family TinyStories checkpoint and its BPE tokenizer; implement the float32 inference path in Rust.
2. Implement RMS normalization, rotary positions, causal attention, grouped key/value heads, SwiGLU, logits, deterministic greedy decoding and seeded sampling. Keep per-run state separate from immutable model weights.
3. Record actual tensor samples, attention matrices and output probabilities on demand. Compare cached and full-prefix inference under the same deterministic workload, with tracing disabled for timing.
4. Validate against an independent NumPy implementation, exercise malformed checkpoints and token/context bounds, then check the real pretrained model.
5. Expose a bounded local HTTP API and CLI. Build a responsive React walkthrough and comparison interface using real API responses.
6. Verify the full browser flow, keyboard/mobile behavior, model download provenance, offline errors, and CI. Publish code, measured evidence, and operating documentation.

## Boundaries

The pretrained weights are downloaded separately and not committed. This project implements inference, not model training. TinyStories is a small story-completion model; it is not an instruction-tuned assistant. Local CPU measurements depend on the machine and workload. Visual traces are numerical observations, not a model's internal reasoning narrative.
