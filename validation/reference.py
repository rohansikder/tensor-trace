"""Independent NumPy forward pass for the legacy llama2.c float32 checkpoint.

This computes an entire sequence with batched matrix operations. It deliberately
does not use the Rust runtime, its KV cache, or its tensor implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_classifier: bool


class Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as source:
            header = source.read(28)
        if len(header) != 28:
            raise ValueError("Incomplete checkpoint header")
        d, hidden, layers, heads, kv_heads, signed_vocab, context = struct.unpack("<7i", header)
        vocab = abs(signed_vocab)
        if min(d, hidden, layers, heads, kv_heads, vocab, context) <= 0:
            raise ValueError("Checkpoint dimensions must be positive")
        if d % heads or heads % kv_heads or (d // heads) % 2:
            raise ValueError("Invalid attention head dimensions")
        self.config = Config(d, hidden, layers, heads, kv_heads, vocab, context, signed_vocab > 0)
        self._data = np.memmap(self.path, mode="r", dtype="<f4", offset=28)
        self._position = 0
        kv_dim = d * kv_heads // heads
        self.embedding = self.take((vocab, d))
        self.attention_norm = self.take((layers, d))
        self.wq = self.take((layers, d, d))
        self.wk = self.take((layers, kv_dim, d))
        self.wv = self.take((layers, kv_dim, d))
        self.wo = self.take((layers, d, d))
        self.ffn_norm = self.take((layers, d))
        self.w1 = self.take((layers, hidden, d))
        self.w2 = self.take((layers, d, hidden))
        self.w3 = self.take((layers, hidden, d))
        self.final_norm = self.take((d,))
        # Legacy exporters include RoPE tables. Like the canonical runtime,
        # compute RoPE directly; the tables are retained for format alignment.
        self.take((context, d // heads // 2))
        self.take((context, d // heads // 2))
        self.classifier = self.embedding if signed_vocab > 0 else self.take((vocab, d))
        if self._position != self._data.size:
            raise ValueError("Checkpoint has unexpected trailing weights")

    def take(self, shape: tuple[int, ...]) -> np.ndarray:
        count = math.prod(shape)
        end = self._position + count
        if end > self._data.size:
            raise ValueError("Checkpoint is shorter than its dimensions require")
        result = self._data[self._position:end].reshape(shape)
        self._position = end
        return result


def rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean_square = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return x / np.sqrt(mean_square + np.float32(1e-5)) * weight


def softmax(scores: np.ndarray) -> np.ndarray:
    maximum = np.max(scores, axis=-1, keepdims=True)
    exponent = np.exp(scores - maximum)
    return exponent / np.sum(exponent, axis=-1, keepdims=True, dtype=np.float32)


def rotary(x: np.ndarray) -> np.ndarray:
    """Rotate adjacent dimensions independently within every attention head."""
    count, _heads, size = x.shape
    frequencies = np.float32(1.0) / np.power(
        np.float32(10000.0), np.arange(0, size, 2, dtype=np.float32) / np.float32(size)
    )
    angles = np.arange(count, dtype=np.float32)[:, None] * frequencies[None, :]
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    result = np.empty_like(x)
    result[..., 0::2] = x[..., 0::2] * cosine - x[..., 1::2] * sine
    result[..., 1::2] = x[..., 0::2] * sine + x[..., 1::2] * cosine
    return result


def forward(model: Checkpoint, tokens: list[int], layer: int = 0, head: int = 0) -> dict:
    c = model.config
    if not tokens or len(tokens) > c.seq_len:
        raise ValueError("Tokens must fit the nonempty model context")
    if any(not isinstance(token, int) or token < 0 or token >= c.vocab_size for token in tokens):
        raise ValueError("Token ID outside vocabulary")
    if not 0 <= layer < c.n_layers or not 0 <= head < c.n_heads:
        raise ValueError("Selected layer or head is outside the model")
    count, head_size = len(tokens), c.dim // c.n_heads
    x = np.array(model.embedding[tokens], dtype=np.float32, copy=True)
    embedding = x.copy()
    selected = None
    for index in range(c.n_layers):
        normalized = rms_norm(x, model.attention_norm[index])
        q = rotary((normalized @ model.wq[index].T).reshape(count, c.n_heads, head_size))
        k = rotary((normalized @ model.wk[index].T).reshape(count, c.n_kv_heads, head_size))
        v = (normalized @ model.wv[index].T).reshape(count, c.n_kv_heads, head_size)
        # Contiguous query-head groups share the corresponding K/V head.
        k = np.repeat(k, c.n_heads // c.n_kv_heads, axis=1)
        v = np.repeat(v, c.n_heads // c.n_kv_heads, axis=1)
        scores = np.einsum("thd,shd->hts", q, k) / np.float32(math.sqrt(head_size))
        future = np.triu(np.ones((count, count), dtype=bool), k=1)
        scores[:, future] = -np.inf
        attention = softmax(scores)
        if index == layer:
            selected = attention[head].copy()
        mixed = np.einsum("hts,shd->thd", attention, v).reshape(count, c.dim)
        x = x + mixed @ model.wo[index].T
        normalized = rms_norm(x, model.ffn_norm[index])
        gate = normalized @ model.w1[index].T
        gate = gate / (np.float32(1.0) + np.exp(-gate))
        hidden = gate * (normalized @ model.w3[index].T)
        x = x + hidden @ model.w2[index].T
    normalized = rms_norm(x, model.final_norm)
    all_logits = normalized @ model.classifier.T
    if not np.isfinite(all_logits).all():
        raise ValueError("Reference produced nonfinite logits")
    assert selected is not None
    return {
        "tokens": tokens,
        "layer": layer,
        "head": head,
        "logits": all_logits[-1].tolist(),
        "attention": selected.tolist(),
        "embedding": embedding[-1].tolist(),
        "greedyToken": int(np.argmax(all_logits[-1])),
        "config": asdict(c),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", default="1,42,7")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = forward(Checkpoint(args.model), [int(x) for x in args.tokens.split(",")], args.layer, args.head)
    text = json.dumps(result, allow_nan=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
