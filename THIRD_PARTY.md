# Model and algorithm provenance

TensorTrace's Rust engine implements inference directly. It does not call a hosted LLM API, Python inference backend, or an external inference runtime.

- **llama2.c**, Andrej Karpathy: the legacy checkpoint layout and tokenizer binary format used for interoperability; [source](https://github.com/karpathy/llama2.c), MIT-licensed code. Its algorithms and documentation are a reference for the architecture and BPE behavior.
- **TinyStories 15M**, Andrej Karpathy: pretrained external weights, downloaded from the pinned [tinyllamas](https://huggingface.co/karpathy/tinyllamas) revision. This project does not claim to train those weights. Model provenance and integrity hashes are in [models/README.md](models/README.md).
- **TinyStories**, Ronen Eldan and Yuanzhi Li: the training dataset underlying the external model; [dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

Third-party Rust and JavaScript dependencies retain their own licenses and are pinned in the corresponding lockfiles. Fonts and icons are installed from their upstream packages.
