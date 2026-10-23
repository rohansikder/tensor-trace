import { useEffect, useRef, useState } from "react";
import {
  ArrowRightIcon,
  ChartLineIcon,
  CheckCircleIcon,
  CircleNotchIcon,
  PlayIcon,
  SquareIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";
import { ComparisonChart } from "./Charts";
import { stream, message, isAbort } from "./api";
import { memory, prettyNumber } from "./App";
import type { Model } from "./types";

interface ModeResult {
  prefillMs: number;
  decodeMsPerToken: number;
  tokensPerSecond: number;
  totalMs: number;
  kvCacheBytes: number;
  generatedTokens: number;
  decodeTokens: number;
  samplesMsPerToken: number[];
}
interface BenchmarkPoint {
  contextLength: number;
  cacheOn: ModeResult;
  cacheOff: ModeResult;
  tokensMatch: boolean;
  speedup: number;
}
interface BenchmarkResult {
  points: BenchmarkPoint[];
  repeats: number;
  maxTokens: number;
  [key: string]: unknown;
}

export function Compare({
  ready,
  model,
  defaultPrompt,
}: {
  ready: boolean;
  model: Model | null;
  defaultPrompt: string;
}) {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [contexts, setContexts] = useState([8, 16, 32]);
  const [tokens, setTokens] = useState(8);
  const [repeats, setRepeats] = useState(3);
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const controller = useRef<AbortController | null>(null);
  const activeJob = useRef(false);
  useEffect(() => () => controller.current?.abort(), []);
  const point = result?.points[selected];
  const [runningSince, setRunningSince] = useState(0);
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    setResult(null);
    setSelected(0);
    setError("");
    setNotice("");
  }, [prompt, contexts, tokens, repeats]);
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(
      () => setSeconds(Math.floor((Date.now() - runningSince) / 1000)),
      1000,
    );
    return () => clearInterval(timer);
  }, [busy, runningSince]);

  function toggleContext(value: number) {
    setContexts((current) =>
      current.includes(value)
        ? current.filter((item) => item !== value)
        : current.length < 4
          ? [...current, value].sort((a, b) => a - b)
          : current,
    );
  }
  async function run() {
    if (activeJob.current || !ready) return;
    activeJob.current = true;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError("");
    setNotice("");
    setSeconds(0);
    setRunningSince(Date.now());
    try {
      const next = await stream<never, BenchmarkResult>(
        "/api/benchmark/stream",
        { prompt, maxTokens: tokens, contextLengths: contexts, repeats },
        abort.signal,
        () => {},
      );
      if (!next)
        throw new Error("The runtime ended without a benchmark result.");
      setResult(next);
      setSelected(0);
    } catch (failure) {
      if (isAbort(failure))
        setNotice(
          "Benchmark cancelled. No incomplete measurements have been added.",
        );
      else setError(message(failure));
    } finally {
      activeJob.current = false;
      setBusy(false);
    }
  }
  const verified = result?.points.every((value) => value.tokensMatch);
  return (
    <>
      <section className="intro compare-intro">
        <span className="eyebrow">INFERENCE WORKBENCH</span>
        <h1>See what the KV cache changes.</h1>
        <p>
          Reuse stored keys and values instead of recomputing the entire prefix.
        </p>
      </section>
      <form
        className="benchmark-controls"
        onSubmit={(event) => {
          event.preventDefault();
          void run();
        }}
      >
        <label className="benchmark-prompt">
          Seed prompt
          <input
            value={prompt}
            disabled={busy}
            onChange={(event) => setPrompt(event.target.value)}
            maxLength={4096}
            placeholder="Once upon a time"
          />
        </label>
        <fieldset className="context-picker" disabled={busy}>
          <legend>Context lengths</legend>
          <div>
            {[8, 16, 32, 64, 128].map((value) => (
              <label
                key={value}
                className={contexts.includes(value) ? "selected" : ""}
              >
                <input
                  type="checkbox"
                  checked={contexts.includes(value)}
                  disabled={
                    (model !== null && value + tokens >= model.config.seqLen) ||
                    (!contexts.includes(value) && contexts.length >= 4)
                  }
                  onChange={() => toggleContext(value)}
                />
                <span>{value}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <label>
          Generated tokens
          <select
            value={tokens}
            disabled={busy}
            onChange={(event) => setTokens(Number(event.target.value))}
          >
            {[4, 8, 16].map((value) => (
              <option value={value} key={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          Repeats
          <select
            value={repeats}
            disabled={busy}
            onChange={(event) => setRepeats(Number(event.target.value))}
          >
            {[1, 2, 3].map((value) => (
              <option value={value} key={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        {busy ? (
          <button
            key="cancel"
            className="secondary-button"
            type="button"
            onClick={(event) => {
              event.preventDefault();
              controller.current?.abort();
            }}
          >
            <SquareIcon size={14} weight="fill" />
            Cancel
          </button>
        ) : (
          <button
            key="run"
            className="primary-button"
            type="submit"
            disabled={!ready || !prompt.trim() || !contexts.length}
          >
            <PlayIcon size={15} weight="fill" />
            Run comparison
          </button>
        )}
      </form>
      <div className="workload-note">
        <span>
          Same input IDs <span>·</span> Greedy decoding <span>·</span> Tracing
          off <span>·</span> Median of repeats
        </span>
        {busy && (
          <span className="running-note" role="status">
            <CircleNotchIcon size={15} className="spin" />
            Measuring both paths · {seconds}s
          </span>
        )}
      </div>
      {error && (
        <div className="inline-message error" role="alert">
          <WarningCircleIcon size={19} />
          <span>{error}</span>
        </div>
      )}
      {notice && (
        <div className="inline-message" role="status">
          {notice}
        </div>
      )}
      <section className="benchmark-results" aria-busy={busy}>
        <div className="benchmark-chart-area">
          <div className="section-title">
            <h2>Decode time per token</h2>
            <span className="subtle-label">
              {result ? "Measured on this CPU" : "No measurements yet"}
            </span>
          </div>
          {result ? (
            <ComparisonChart
              points={result.points.map((value) => ({
                contextTokens: value.contextLength,
                cacheOnMs: value.cacheOn.decodeMsPerToken,
                cacheOffMs: value.cacheOff.decodeMsPerToken,
              }))}
            />
          ) : (
            <div className="benchmark-empty">
              <ChartLineIcon size={49} weight="duotone" />
              <h3>Run the same workload two ways.</h3>
              <p>
                Compare cached and uncached inference using identical input
                tokens. The chart fills with actual measurements from this
                machine.
              </p>
              {busy && (
                <span className="subtle-label">
                  <CircleNotchIcon size={15} className="spin" />
                  Warm-up, alternating runs, then verification
                </span>
              )}
            </div>
          )}
        </div>
        <aside className="benchmark-table-area">
          <div className="section-title">
            <h2>Compare a context</h2>
          </div>
          {point ? (
            <>
              <label className="context-select">
                <span className="sr-only">Measured context length</span>
                <select
                  value={selected}
                  onChange={(event) => setSelected(Number(event.target.value))}
                >
                  {result!.points.map((value, i) => (
                    <option value={i} key={value.contextLength}>
                      {value.contextLength} context tokens
                    </option>
                  ))}
                </select>
              </label>
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th scope="col">Measurement</th>
                    <th scope="col">Cache off</th>
                    <th scope="col">Cache on</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">Decode / token</th>
                    <td>
                      {prettyNumber(point.cacheOff.decodeMsPerToken, 2)} ms
                    </td>
                    <td>
                      {prettyNumber(point.cacheOn.decodeMsPerToken, 2)} ms
                    </td>
                  </tr>
                  <tr>
                    <th scope="row">Prefill</th>
                    <td>{prettyNumber(point.cacheOff.prefillMs, 2)} ms</td>
                    <td>{prettyNumber(point.cacheOn.prefillMs, 2)} ms</td>
                  </tr>
                  <tr>
                    <th scope="row">Reusable K/V</th>
                    <td>{memory(point.cacheOff.kvCacheBytes)}</td>
                    <td>{memory(point.cacheOn.kvCacheBytes)}</td>
                  </tr>
                  <tr>
                    <th scope="row">Decode tokens</th>
                    <td>{point.cacheOff.decodeTokens}</td>
                    <td>{point.cacheOn.decodeTokens}</td>
                  </tr>
                </tbody>
              </table>
              <div
                className={`token-match ${point.tokensMatch ? "passed" : "failed"}`}
              >
                <CheckCircleIcon size={18} weight="fill" />
                <span>
                  {point.tokensMatch
                    ? "Generated token IDs match exactly"
                    : "Token equality check failed"}
                </span>
              </div>
              <p className="detail-caption">
                Decode timing excludes the first prediction and prompt prefill.
                Stored K/V is the reusable payload, not total process memory.
                The uncached path still uses transient working memory.
              </p>
            </>
          ) : (
            <div className="table-empty">
              <span>Cache off</span>
              <span>Cache on</span>
              <p>
                Timings and memory will appear after a completed comparison.
              </p>
            </div>
          )}
        </aside>
      </section>
      <section className="cache-explanation">
        <h2>Where the work goes</h2>
        <div className="cache-row">
          <div>
            <strong>Cache off</strong>
            <p>Each prediction processes the entire growing prefix.</p>
          </div>
          <div className="cache-process">
            <span className="work-block">Prefix embeddings</span>
            <ArrowRightIcon size={18} />
            <span className="work-block">All context positions</span>
            <ArrowRightIcon size={18} />
            <span className="work-block result-block">Next token</span>
          </div>
        </div>
        <div className="cache-row">
          <div>
            <strong>Cache on</strong>
            <p>
              Stored keys and values are reused while the new position is
              computed.
            </p>
          </div>
          <div className="cache-process">
            <span className="work-block reusable">Stored K/V</span>
            <ArrowRightIcon size={18} />
            <span className="work-block">New position + attention</span>
            <ArrowRightIcon size={18} />
            <span className="work-block result-block">Next token</span>
          </div>
        </div>
        <p className="detail-caption">
          More context uses more cache memory. Attention for each new token
          still reads the available context.
        </p>
      </section>
      <footer className="workspace-footer">
        <span>
          Local CPU measurements describe this machine, model, and workload.
        </span>
        <span>
          {result ? (
            <>
              {result.points.length} contexts measured{" "}
              {verified
                ? "· token equality verified"
                : "· inspect token equality"}
            </>
          ) : (
            "Both paths are warmed before timed runs."
          )}
        </span>
      </footer>
    </>
  );
}
