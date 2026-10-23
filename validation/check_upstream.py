"""Optional three-implementation check using a pinned canonical llama2.c source.

The upstream C source is downloaded to a temporary directory, hash checked,
compiled without fast-math, executed, and removed. It is not vendored or used
by the Rust application. A C compiler and network access are needed unless
--source points to the exact pinned source already downloaded locally.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import urllib.request

from check import compare_array, inspect
from reference import Checkpoint, forward

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "350e04fe35433e6d2941dce5a1f53308f87058eb"
SOURCE_URL = f"https://raw.githubusercontent.com/karpathy/llama2.c/{SOURCE_COMMIT}/run.c"
SOURCE_SHA256 = "9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd"
WRAPPER = r'''
#define main canonical_generation_main
#include "upstream.c"
#undef main

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    char *ids = strdup(argv[2]);
    float *logits = NULL;
    int position = 0;
    for (char *part = strtok(ids, ","); part; part = strtok(NULL, ",")) {
        int token = atoi(part);
        if (token < 0 || token >= transformer.config.vocab_size || position >= transformer.config.seq_len) return 3;
        logits = forward(&transformer, token, position++);
    }
    if (!logits) return 4;
    printf("{\"logits\":[");
    for (int i = 0; i < transformer.config.vocab_size; i++) {
        if (i) putchar(',');
        printf("%.9g", logits[i]);
    }
    printf("]}\n");
    free(ids);
    free_transformer(&transformer);
    return 0;
}
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/stories15M.bin")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/tokenizer.bin")
    parser.add_argument("--binary", type=Path, help="Also compare the Rust inspector")
    parser.add_argument("--source", type=Path, help="Previously downloaded exact pinned run.c")
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--output", type=Path, default=ROOT / "build/canonical-numeric-validation.json")
    args = parser.parse_args()
    compiler = shutil.which(args.cc)
    if not compiler:
        parser.error(f"C compiler unavailable: {args.cc}")
    report = {"schemaVersion": 1, "checkedAt": datetime.now(timezone.utc).isoformat(), "sourceUrl": SOURCE_URL, "sourceCommit": SOURCE_COMMIT, "sourceSha256": SOURCE_SHA256, "compilerFlags": ["-O2", "-lm"], "method": "Canonical C, independent whole-sequence NumPy, and optionally Rust full vocabulary logits", "absoluteTolerance": 3e-4, "relativeTolerance": 3e-4, "status": "running", "checks": []}
    try:
        checkpoint = Checkpoint(args.model)
        report["modelSha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
        if args.source:
            source = args.source.read_bytes()
        else:
            request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "TensorTrace-numeric-validation"})
            with urllib.request.urlopen(request, context=ssl.create_default_context(), timeout=30) as response:
                source = response.read(2_000_001)
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("Canonical source SHA-256 mismatch; refusing to compile it")
        cases = [[1], [1, 42, 7, 300], [1, 100, 200, 300, 400, 500, 600, 700]]
        with tempfile.TemporaryDirectory(prefix="tensor-trace-reference-") as directory:
            working = Path(directory)
            (working / "upstream.c").write_bytes(source)
            (working / "inspect.c").write_text(WRAPPER)
            executable = working / "inspect"
            subprocess.run([compiler, "-O2", "inspect.c", "-o", str(executable), "-lm"], cwd=working, check=True, capture_output=True, text=True, timeout=60)
            for tokens in cases:
                expected = forward(checkpoint, tokens)
                process = subprocess.run([str(executable), str(args.model.resolve()), ",".join(map(str, tokens))], capture_output=True, text=True, check=True, timeout=60)
                canonical = json.loads(process.stdout)
                comparison = compare_array(canonical["logits"], expected["logits"], "C versus NumPy logits", 3e-4, 3e-4)
                result = {"tokens": tokens, "canonicalVsNumpy": comparison, "status": "passed"}
                if args.binary:
                    case = {"tokens": tokens, "layer": 0, "head": 0}
                    for cached in (True, False):
                        actual = inspect(args.binary, args.model, args.tokenizer, case, cached)
                        result["canonicalVsRustCached" if cached else "canonicalVsRustUncached"] = compare_array(actual["logits"], canonical["logits"], "Rust versus C logits", 3e-4, 3e-4)
                report["checks"].append(result)
                print(f"PASS canonical C / NumPy: {len(tokens)} tokens; max logit error {comparison['maxAbsoluteError']:.7g}")
        report["status"] = "passed"
    except (AssertionError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        detail = error.stderr.strip()[-2000:] if isinstance(error, subprocess.CalledProcessError) and error.stderr else ""
        report.update(status="failed", error=f"{error}\n{detail}".strip())
        print(f"FAIL: {report['error']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
