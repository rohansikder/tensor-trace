"""Rebuild the small, untrained numeric fixtures and their NumPy goldens.

These fixtures exercise the real checkpoint format, including grouped-query
attention and both tied and untied classifiers. They do not generate meaningful
language and must never be presented as a trained demo model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from reference import Checkpoint, forward

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
SEED = 20261023
DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB, CONTEXT = 16, 32, 2, 4, 2, 512, 32
CASES = [
    {"tokens": [1], "layer": 0, "head": 0},
    {"tokens": [1, 77], "layer": 1, "head": 3},
    {"tokens": [1, 7, 29, 4], "layer": 0, "head": 1},
    {"tokens": [1, 19, 3, 300, 65, 2, 9], "layer": 1, "head": 2},
    {"tokens": [1, 5, 28, 97, 54, 12, 23, 51, 6, 8, 34, 321, 15, 127, 17], "layer": 1, "head": 3},
]


def write_model(path: Path, shared: bool) -> None:
    rng = np.random.default_rng(SEED if shared else SEED + 1)
    parts = []

    def matrix(shape: tuple[int, ...], scale: float) -> None:
        parts.append(rng.normal(0.0, scale, shape).astype("<f4").tobytes())

    def norm(shape: tuple[int, ...]) -> None:
        parts.append((1.0 + rng.normal(0.0, 0.03, shape)).astype("<f4").tobytes())

    kv_dim = DIM * KV_HEADS // HEADS
    matrix((VOCAB, DIM), 0.2)
    norm((LAYERS, DIM))
    matrix((LAYERS, DIM, DIM), DIM ** -0.5)
    matrix((LAYERS, kv_dim, DIM), DIM ** -0.5)
    matrix((LAYERS, kv_dim, DIM), DIM ** -0.5)
    matrix((LAYERS, DIM, DIM), DIM ** -0.5)
    norm((LAYERS, DIM))
    matrix((LAYERS, HIDDEN, DIM), DIM ** -0.5)
    matrix((LAYERS, DIM, HIDDEN), HIDDEN ** -0.5)
    matrix((LAYERS, HIDDEN, DIM), DIM ** -0.5)
    norm((DIM,))
    frequencies = np.float32(1.0) / np.power(
        np.float32(10000.0), np.arange(0, DIM // HEADS, 2, dtype=np.float32) / np.float32(DIM // HEADS)
    )
    angles = np.arange(CONTEXT, dtype=np.float32)[:, None] * frequencies
    parts.extend([np.cos(angles).astype("<f4").tobytes(), np.sin(angles).astype("<f4").tobytes()])
    if not shared:
        matrix((VOCAB, DIM), DIM ** -0.5)
    header = struct.pack("<7i", DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB if shared else -VOCAB, CONTEXT)
    path.write_bytes(header + b"".join(parts))


def write_tokenizer(path: Path) -> None:
    vocabulary = ["<unk>", "<s>", "</s>"]
    vocabulary.extend(f"<0x{i:02X}>" for i in range(256))
    vocabulary.extend(chr(i) for i in range(32, 127))
    vocabulary.extend(["Th", "The", " c", " ca", " cat", " s", " sa", " sat", " o", " on", " t", " th", " the", " m", " ma", " mat"])
    vocabulary.extend(f"<unused_{i}>" for i in range(VOCAB - len(vocabulary)))
    assert len(vocabulary) == VOCAB and len(set(vocabulary)) == VOCAB
    encoded = [piece.encode("utf-8") for piece in vocabulary]
    data = [struct.pack("<i", max(map(len, encoded)))]
    for index, piece in enumerate(encoded):
        score = -10.0 if index < 259 else float(len(piece))
        data.extend([struct.pack("<fi", score, len(piece)), piece])
    path.write_bytes(b"".join(data))


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    result = {"schemaVersion": 1, "seed": SEED, "description": "Untrained synthetic numeric fixtures, not language models", "models": []}
    for shared in (True, False):
        path = FIXTURES / ("tiny-shared.bin" if shared else "tiny-unshared.bin")
        write_model(path, shared)
        checkpoint = Checkpoint(path)
        result["models"].append({
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cases": [forward(checkpoint, **case) for case in CASES],
        })
    write_tokenizer(FIXTURES / "tokenizer.bin")
    (FIXTURES / "goldens.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Rebuilt shared/unshared fixtures, tokenizer, and independent NumPy goldens.")


if __name__ == "__main__":
    main()
