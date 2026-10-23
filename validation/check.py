"""Compare Rust inspector output with an independent full-sequence NumPy pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from reference import Checkpoint, forward

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def compare_array(actual, expected, name: str, absolute: float, relative: float) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all():
        raise AssertionError(f"{name}: Rust returned a nonfinite value")
    difference = np.abs(actual - expected)
    if not np.allclose(actual, expected, atol=absolute, rtol=relative):
        position = np.unravel_index(int(np.argmax(difference)), difference.shape)
        raise AssertionError(
            f"{name}: max absolute error {float(difference.max()):.8g} at {position}; "
            f"Rust={actual[position]:.8g}, NumPy={expected[position]:.8g} "
            f"(atol={absolute}, rtol={relative})"
        )
    return {"shape": list(actual.shape), "maxAbsoluteError": float(difference.max(initial=0.0))}


def inspect(binary: Path, model: Path, tokenizer: Path, case: dict, cached: bool) -> dict:
    command = [
        str(binary.resolve()), "inspect", "--model", str(model.resolve()),
        "--tokenizer", str(tokenizer.resolve()),
        "--tokens", ",".join(map(str, case["tokens"])),
        "--cache", "on" if cached else "off", "--layer", str(case["layer"]), "--head", str(case["head"]),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise ValueError(f"Rust inspector exited {result.returncode}: {result.stderr.strip()[-2000:]}")
    return json.loads(result.stdout)


def model_cases(path: Path, tiny: bool) -> list[dict]:
    if tiny:
        source = json.loads((FIXTURES / "goldens.json").read_text())
        return next(model["cases"] for model in source["models"] if model["file"] == path.name)
    config = Checkpoint(path).config
    # Explicit token IDs isolate numeric inference from tokenizer behavior.
    sequences = [[1], [1, 42, 7, 300], [1, 100, 200, 300, 400, 500, 600, 700]]
    return [{"tokens": sequence, "layer": index % config.n_layers, "head": (index * 2) % config.n_heads} for index, sequence in enumerate(sequences)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/tensor-trace")
    parser.add_argument("--model", type=Path, help="Real pretrained checkpoint; omit to test both tiny fixtures")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "build/numeric-validation.json")
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=3e-4)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"Build the Rust binary first: {args.binary}")
    models = [args.model] if args.model else [FIXTURES / "tiny-shared.bin", FIXTURES / "tiny-unshared.bin"]
    tokenizer = args.tokenizer or (ROOT / "models/tokenizer.bin" if args.model else FIXTURES / "tokenizer.bin")
    report = {"schemaVersion": 1, "checkedAt": datetime.now(timezone.utc).isoformat(), "method": "Independent batched NumPy full-sequence forward pass versus Rust cached and uncached inference", "absoluteTolerance": args.atol, "relativeTolerance": args.rtol, "status": "running", "checks": []}
    try:
        for path in models:
            checkpoint = Checkpoint(path)
            model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if args.model is None:
                manifest = json.loads((FIXTURES / "goldens.json").read_text())
                recorded = next(item for item in manifest["models"] if item["file"] == path.name)
                if model_hash != recorded["sha256"]:
                    raise AssertionError(f"Fixture hash does not match its committed golden data: {path.name}")
            for case in model_cases(path, tiny=args.model is None):
                expected = forward(checkpoint, case["tokens"], case["layer"], case["head"])
                if args.model is None:
                    compare_array(expected["logits"], case["logits"], "committed golden logits", 2e-5, 2e-5)
                    compare_array(expected["attention"], case["attention"], "committed golden attention", 2e-5, 2e-5)
                cached_result = None
                for cached in (True, False):
                    actual = inspect(args.binary, path, tokenizer, case, cached)
                    logits = compare_array(actual["logits"], expected["logits"], "logits", args.atol, args.rtol)
                    attention = compare_array(actual["attention"], expected["attention"], "attention", args.atol, args.rtol)
                    matrix = np.asarray(actual["attention"])
                    if not np.all(matrix[np.triu_indices(len(case["tokens"]), 1)] == 0):
                        raise AssertionError("Future-token attention must be exactly masked to zero")
                    compare_array(matrix.sum(axis=1), np.ones(len(matrix)), "attention row sums", 1e-5, 1e-5)
                    greedy = int(np.argmax(actual["logits"]))
                    if greedy != expected["greedyToken"]:
                        raise AssertionError("Rust and NumPy disagree on greedy next token")
                    equivalence = None
                    if cached:
                        cached_result = actual
                    else:
                        equivalence = compare_array(actual["logits"], cached_result["logits"], "cached/uncached logits", args.atol, args.rtol)
                    report["checks"].append({"model": path.name, "modelSha256": model_hash, "tokens": case["tokens"], "layer": case["layer"], "head": case["head"], "cacheEnabled": cached, "greedyToken": greedy, "logits": logits, "attention": attention, "cacheEquivalence": equivalence, "status": "passed"})
                    print(f"PASS {path.name}: {len(case['tokens'])} tokens, layer {case['layer']}, head {case['head']}, cache {'on' if cached else 'off'}; max logit error {logits['maxAbsoluteError']:.7g}", flush=True)
        report["status"] = "passed"
    except (AssertionError, ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=str(error))
        print(f"FAIL: {error}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
