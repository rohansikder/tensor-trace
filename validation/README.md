# Numeric validation

The Rust runtime is checked against a separately implemented NumPy reference.
The reference uses whole-sequence matrix multiplication, `einsum`, and a causal
mask. It does not call Rust or reuse the engine's incremental KV cache logic.

Both implementations read the documented legacy `llama2.c` float32 checkpoint
layout: a seven-integer header, embedding and layer weights, reserved rotary
tables, and an optional untied classifier. The reference computes RMSNorm,
rotary positions, grouped-query causal attention, residuals, SwiGLU, and the
final vocabulary logits.

## Run the checks

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r validation/requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s validation -p 'test_*.py'
cargo build --release --bin tensor-trace
OPENBLAS_NUM_THREADS=1 .venv/bin/python validation/check.py
```

The default check uses two committed, deterministic **untrained numeric
fixtures**, not the pretrained demonstration model. Each fixture has two
layers, four query heads, two K/V heads, sixteen hidden dimensions, and a
512-token vocabulary. One ties the classifier to the embeddings; the other
uses a separate classifier. Five token sequences inspect multiple layers and
heads with caching both enabled and disabled: twenty Rust/reference checks.
The fixtures and their complete NumPy golden logits can be rebuilt with:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python validation/make_fixtures.py
```

Seed `20261023`, the pinned NumPy version, fixture SHA-256 hashes, and full
golden arrays are recorded in `fixtures/goldens.json`. Regeneration is an
explicit maintenance action, not part of normal validation. No fixture is used
as a fallback language model in the application.

For the actual downloaded pretrained model:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python validation/check.py \
  --model models/stories15M.bin \
  --tokenizer models/tokenizer.bin \
  --output build/pretrained-numeric-validation.json
```

This checks three sequences, including multiple rotary positions, with both
cache modes. It compares all 32,000 final vocabulary logits and every cell of
the selected layer/head attention matrix. The report records the exact model
hash, token IDs, layer/head, cache setting, maximum absolute errors, and pass
or fail status. The CLI exits unsuccessfully if any check fails.

The default tolerance is `atol=3e-4, rtol=3e-4`. NumPy's matrix multiplication
and Rust's scalar/vectorized accumulations have different floating-point
reduction orders, so exact bit equality is not required. Every comparison also
requires the same greedy next token, exactly zero attention to future tokens,
normalized attention rows, and cache-on/cache-off logit agreement.

## Optional canonical C cross-check

A third implementation can be checked as well:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python validation/check_upstream.py \
  --binary target/release/tensor-trace
```

This optional command requires a C compiler and internet access. It downloads
the canonical [`karpathy/llama2.c` `run.c`](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c),
pinned to commit `350e04fe35433e6d2941dce5a1f53308f87058eb` and SHA-256
`9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd`.
Only source matching that hash is compiled. The program is built without
fast-math in a temporary directory, exposes its complete final logits, and is
removed afterward. Source and model hashes, compilation flags, and all errors
are recorded in `build/canonical-numeric-validation.json`. The C source stays
upstream and is not part of the Rust application or this repository.

The same three pretrained token sequences compare canonical C with NumPy and,
when `--binary` is supplied, with both Rust cache modes. This cross-check covers
logits; the more detailed NumPy check also covers attention matrices. To run
without downloading the source again, use `--source /path/to/pinned/run.c`;
its hash is still verified before compilation.

## Interpretation and limits

These checks establish numerical agreement for the recorded inputs and model
hash. They do not prove every possible input, every model architecture, text
quality, training correctness, GPU behavior, or a performance claim. Explicit
token IDs isolate the forward pass from tokenizer behavior; tokenizer tests are
separate. Reported validation durations are not inference benchmarks.

The pretrained model and tokenizer provenance is documented in
[`models/README.md`](../models/README.md). NumPy is used only for independent
validation; the application performs inference in Rust.
