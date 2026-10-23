# Recorded evidence

These files are snapshots from actual execution on 23 October 2026. Fresh runs
write to the ignored `build/` directory; they do not silently replace the
committed evidence. Screenshots in [../screenshots](../screenshots/) were captured
from the same working application, not from a design mock.

## Cache comparison

[benchmark.json](benchmark.json) contains the response captured from the release
Rust server by the browser check. The seed prompt was `Once upon a time`.
Each exact context length was constructed from repeated prompt token IDs, with
one BOS token. Both modes generated eight greedy tokens. Tracing was off; both
paths were warmed once per context, followed by three measured repeats with
alternating execution order. The first prediction belongs to prefill; seven
later predictions form the decode measurements.

| Context tokens | Cache off, ms/decode token | Cache on, ms/decode token | Ratio off/on |
| ---: | ---: | ---: | ---: |
| 8 | 59.42 | 5.27 | 11.29× |
| 16 | 103.69 | 5.51 | 18.82× |
| 32 | 183.38 | 5.35 | 34.30× |

All generated token IDs matched exactly between modes. The JSON preserves raw
samples, medians, prefill times, generated counts, and reusable K/V payload.
Payload is active retained keys and values, not process RSS or allocator capacity.

The environment was a Linux container on an AMD EPYC 9V74, with three visible
logical CPUs and a cgroup quota equivalent to two CPUs. The model forward pass
uses one thread. The release build used Rust 1.99.1 and its default target CPU,
without custom `RUSTFLAGS`. See [environment.json](environment.json) for the
toolchain, executable hash, and source-file hashes.
The measurements include ordinary container scheduling noise and only three
repeats. They describe this implementation and workload, not a general claim
about other runtimes, machines, or large models.

To measure again:

```bash
target/release/tensor-trace bench --context-lengths 8,16,32 --max-tokens 8 --repeats 3 > build/benchmark.json
```

## Numerical agreement

| Report | Coverage | Result |
| --- | --- | --- |
| [Tiny fixtures](tiny-numeric-validation.json) | 20 cache-on/off checks across shared/untied classifiers and grouped K/V heads | Passed |
| [Pretrained checkpoint](pretrained-numeric-validation.json) | 6 checks comparing all 32,000 final logits and selected attention matrices | Passed |
| [Canonical C](canonical-numeric-validation.json) | 3 contexts comparing canonical C, NumPy, and both Rust cache modes | Passed |

The largest recorded pretrained logit difference was approximately `1.84e-5`
between Rust and NumPy, and `3.36e-5` between Rust and canonical C. Each numeric
report records its tolerances, model SHA-256, explicit token IDs, and errors.
The [validation guide](../../validation/README.md) explains the independent
reference, commands, and limits of these checks. The application uses the real
pretrained checkpoint; the tiny untrained fixtures are validation assets only.

## Browser verification

[browser-verification.json](browser-verification.json) records twelve passing
checks against the release runtime, with no browser exceptions or console
errors. It covers readiness/empty states, all six inspection stages, selected
layers/heads, keyboard navigation, streamed output, settings invalidation,
seeded generation, structured errors, measured cache comparison, generation
and benchmark cancellation, mobile widths, and missing-model behavior.

The desktop viewport was 1487×1058 and the mobile viewport was 390×844. Full-page
screenshots are in [../screenshots](../screenshots/). To reproduce all checks,
start the regular server on port 8787 and a second unavailable server:

```bash
target/release/tensor-trace --model build/intentionally-missing-model.bin serve --port 8788
```

In another terminal:

```bash
.venv/bin/pip install -r scripts/requirements-browser.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/browser_check.py --unavailable-url http://127.0.0.1:8788
```

`BROWSER_EXECUTABLE` can select a system Chromium executable. CI reruns the
browser suite and numerical reference and uploads fresh evidence as artifacts.
The source hashes identify the files used for this local snapshot; the snapshot
was captured from the implementation working tree before its initial commit.
