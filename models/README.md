# External model assets

Run `python3 scripts/fetch_model.py` from the repository root. This fetches approximately 61 MB and verifies every file against its pinned SHA-256. It requires network access only during setup; inference runs entirely on the CPU without a hosted model API.

| Asset | Source revision | SHA-256 |
| --- | --- | --- |
| `stories15M.bin` | `karpathy/tinyllamas` · `0bd21da7698eaf29a0d7de3992de8a46ef624add` | `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a` |
| `tokenizer.bin` | `karpathy/llama2.c` · `350e04fe35433e6d2941dce5a1f53308f87058eb` | `50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361` |

The model is Andrej Karpathy's small Llama-family model trained on TinyStories. Its checkpoint has dimension 288, hidden dimension 768, six layers, six attention heads, a 32,000-token vocabulary, and a 256-token context. It completes short stories; it is not an instruction-tuned chat assistant.

The upstream [model card](https://huggingface.co/karpathy/tinyllamas/blob/0bd21da7698eaf29a0d7de3992de8a46ef624add/README.md) declares an MIT license. The tokenizer is supplied separately by [llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb). External assets are downloaded from their original sources and are not redistributed in this Git repository. The project license covers TensorTrace's code; upstream asset terms remain with their respective sources.

`python3 scripts/fetch_model.py --verify-only` performs an offline integrity check. Alternative files can be passed to the runtime explicitly; compatibility is limited to the supported legacy float32 checkpoint and binary tokenizer format, not arbitrary Hugging Face model directories or GGUF.
