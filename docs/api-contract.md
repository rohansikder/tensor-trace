# Local runtime API

The Rust process serves `/api` and the compiled web console on the same origin.
All JSON keys use camelCase. There is no remote inference service or API key.

## Health and model

- `GET /api/health`: always responds with readiness (`status: "ready"` or `"missing_model"`), engine name, and model-loading error when unavailable.
- `GET /api/model`: loaded checkpoint metadata; returns 503 with an actionable error when weights/tokenizer are missing.
- `POST /api/tokenize`: `{ "text": "Once upon a time" }`; returns token IDs and decoded pieces.

## Generation

`POST /api/generate`, or `POST /api/generate/stream` for SSE:

```json
{
  "prompt": "Once upon a time",
  "maxTokens": 16,
  "temperature": 0,
  "topP": 0.9,
  "seed": 42,
  "cache": true,
  "layer": 0,
  "head": 0,
  "trace": true
}
```

The selected layer/head controls captured activations and attention; it does not modify inference. Generation includes actual checkpoint results and timings. Limits: at most 64 generated tokens, 4 KiB prompt text, model context bound, temperatures from 0 to 2 and top-p in (0,1]. At most one expensive job executes at once by default. Further requests receive 429 rather than accumulating CPU work.

SSE event types:

- `token`: one actual `StepTrace`, emitted after computing that generated token.
- `done`: the complete `GenerationResult` with the same actual tokens and traces.
- `error`: `{ "error": "...", "code": "..." }` for a runtime error after streaming starts.

The HTTP request must be validated before opening the stream. A dropped stream requests cancellation at the next token boundary. Runs are capped at 120 seconds, with cancellation checked at token boundaries. A single forward pass is not interrupted mid-matrix-operation.

## Benchmark

`POST /api/benchmark`:

```json
{
  "prompt": "Once upon a time",
  "maxTokens": 8,
  "contextLengths": [8, 16, 32],
  "repeats": 3
}
```

Benchmarking uses identical input token IDs and greedy generated sequences for cached and uncached execution, with tracing disabled. The seed prompt is repeated as token IDs to construct each exact context length. It warms both paths, alternates order across repeats, checks output-token equality, and reports medians of real timings and exact active reusable KV payload bytes (not allocated buffer capacity, total RAM or peak RAM). It is local CPU evidence, not a cross-machine performance claim.

`POST /api/benchmark/stream` accepts the same request and sends the final benchmark as an SSE `done` event, or a structured `error` event. The browser uses this endpoint so Cancel closes the stream and requests that CPU work stop at the next generated-token boundary. No partial benchmark is reported as a completed measurement. The JSON endpoint is bounded by the same deadline but does not provide this disconnect-driven cancellation contract.

### Response shapes

`GET /api/model` returns:

```text
{name, architecture, config:{dim,hiddenDim,nLayers,nHeads,nKvHeads,vocabSize,seqLen},
 parameterCount,modelBytes,tokenizerBytes,engine}
```

`GenerationResult` is:

```text
{text,generatedText,promptTokens:[{id,text}],generatedTokens:[{id,text}],steps:[StepTrace],
 stats:{promptTokens,generatedTokens,decodeTokens,prefillMs,decodeMs,forwardDecodeMs,timeToFirstTokenMs,samplingMs,totalMs,
        tokensPerSecond,decodeMsPerToken,kvCacheBytes,cacheEnabled,contextCapacity,stoppedReason,seed}}
```

`StepTrace` is:

```text
{index,position,token:{id,text},text,contextTokens:[{id,text}],attention:number[][],
 embedding:number[],hiddenState:number[],query:number[],key:number[],value:number[],
 attentionOutput:number[],feedForward:number[],
 topTokens:[{id,text,probability,logit}],entropy,selectedProbability,layer,head,
 timing:{forwardMs,samplingMs,totalMs}}
```

`attention` contains the actual selected head's causal attention matrix, query rows by key columns. `topTokens` and `selectedProbability` describe the original model softmax, before temperature/top-p transformations. `text` in each token event is accumulated generated text, not the prompt. Statistics distinguish first-token prefill from subsequent decode steps: `decodeTokens = generatedTokens - 1`. `decodeMs` includes forward computation plus sampling/probability-summary computation for tokens after the first; `tokensPerSecond` is steady decode throughput with the first token excluded. `forwardDecodeMs` exposes only matrix/forward computation. `prefillMs` is the first forward computation; `timeToFirstTokenMs` adds that first token's sampling/summary computation. Transport time, trace serialization and browser rendering are not included. `kvCacheBytes` counts active reusable K/V entries, not buffer allocation, vector capacity, process RAM, or peak memory. Both modes allocate computation workspace, and uncached execution still needs transient K/V entries. Zero means no K/V payload is reused between decode steps, not zero memory consumption.

`BenchmarkResult` is:

```text
{model,points:[{contextLength,cacheOn:Measurement,cacheOff:Measurement,tokensMatch,speedup}],
 repeats,maxTokens,elapsedMs,method,tracing:false,promptConstruction}
```

`Measurement` is `{prefillMs,timeToFirstTokenMs,forwardDecodeMs,decodeMsPerToken,tokensPerSecond,totalMs,kvCacheBytes,generatedTokens,decodeTokens,samplesMsPerToken}`. Each value is the median across measured repeats, except exact counts/active reusable cache payload and raw decode samples. `speedup` is uncached median decode time divided by cached median decode time. The first sampled token is produced by prefill and is excluded from decode timing. Benchmarks disable EOS stopping to preserve the requested output-token count.

Benchmark request limits are 2–16 output tokens, 1–3 repeats, 1–4 unique context lengths each 2–128, all within the loaded model capacity. A cumulative workload bound also limits expensive combinations.

## Errors

Errors have `{ "error": "Human-readable message", "code": "invalid_request" }`. Relevant HTTP statuses: 400 invalid request, 404 unknown API route, 429 runtime busy, 503 missing model. Unknown `/api/...` routes never fall back to frontend HTML.
