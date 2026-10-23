import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRightIcon,
  ArrowClockwiseIcon,
  ArrowUpRightIcon,
  BracketsSquareIcon,
  ChartLineIcon,
  CircleNotchIcon,
  CpuIcon,
  GraphIcon,
  InfoIcon,
  PlayIcon,
  SlidersHorizontalIcon,
  SquareIcon,
  StackIcon,
  WarningCircleIcon,
  XIcon,
} from "@phosphor-icons/react";
import { request, stream, isAbort, message } from "./api";
import { AttentionChart, VectorChart } from "./Charts";
import { Compare } from "./Compare";
import type {
  Model,
  Health,
  StepTrace,
  Generation,
  GenerationInput,
  Token,
  Candidate,
} from "./types";

const stages = [
  "Text",
  "Tokens",
  "Vectors",
  "Transformer",
  "Scores",
  "Next token",
];
const stageDescriptions = [
  "Start with text. The engine adds a beginning-of-sequence token before reading your prompt.",
  "A byte-pair tokenizer maps text pieces to integer IDs from the model’s vocabulary.",
  "Each token ID selects a learned embedding vector. These values come from the loaded checkpoint.",
  "Causal attention mixes earlier context; a feed-forward network transforms each position.",
  "The final hidden state is projected into vocabulary scores, called logits. Softmax turns them into probabilities.",
  "Greedy decoding takes the highest-scoring token. With temperature enabled, the sampler draws from a filtered distribution.",
];

export const prettyNumber = (value: number, digits = 1) =>
  new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(value);
export const memory = (bytes: number) =>
  bytes >= 1024 * 1024
    ? `${prettyNumber(bytes / 1024 / 1024, 2)} MiB`
    : bytes >= 1024
      ? `${prettyNumber(bytes / 1024, 1)} KiB`
      : `${bytes} B`;
export const tokenText = (token: Token) =>
  token.id === 1 && !token.text
    ? "<BOS>"
    : token.id === 2 && !token.text
      ? "<EOS>"
      : token.text.replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/^ /, "·") ||
        "∅";

export default function App() {
  const [view, setView] = useState<"walkthrough" | "compare">("walkthrough");
  const [model, setModel] = useState<Model | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const [loading, setLoading] = useState(true);
  const [about, setAbout] = useState(false);
  const [prompt, setPrompt] = useState("Once upon a time");
  const [steps, setSteps] = useState<StepTrace[]>([]);
  const [selectedStep, setSelectedStep] = useState(0);
  const [phase, setPhase] = useState(0);
  const [result, setResult] = useState<Generation | null>(null);
  const [busy, setBusy] = useState<"step" | "run" | "inspect" | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [maxTokens, setMaxTokens] = useState(16);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(0.9);
  const [seed, setSeed] = useState(42);
  const [cache, setCache] = useState(true);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [runInput, setRunInput] = useState<GenerationInput | null>(null);
  const abort = useRef<AbortController | null>(null);
  const aboutTrigger = useRef<HTMLButtonElement | null>(null);
  const activeJob = useRef(false);
  const alive = useRef(true);

  const connect = useCallback(async () => {
    setLoading(true);
    setConnectionError("");
    try {
      const next = await request<Health>("/api/health");
      if (!alive.current) return;
      setHealth(next);
      if (next.status === "ready") setModel(await request<Model>("/api/model"));
      else setModel(null);
    } catch (failure) {
      if (alive.current) {
        setHealth(null);
        setModel(null);
        setConnectionError(message(failure));
      }
    } finally {
      if (alive.current) setLoading(false);
    }
  }, []);
  useEffect(() => {
    alive.current = true;
    void connect();
    return () => {
      alive.current = false;
      abort.current?.abort();
    };
  }, [connect]);
  useEffect(() => {
    if (!about) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setAbout(false);
      if (event.key === "Tab") {
        const elements = document.querySelectorAll<HTMLElement>(
          ".about-dialog button, .about-dialog a[href]",
        );
        const first = elements[0];
        const last = elements[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("keydown", close);
      aboutTrigger.current?.focus();
    };
  }, [about]);

  const ready = model !== null && health?.status === "ready";
  const current = steps[selectedStep] ?? null;
  const inputTokens = current?.contextTokens ?? [];
  const runDone = steps.length > 0;

  function clearRun() {
    abort.current?.abort();
    setSteps([]);
    setSelectedStep(0);
    setResult(null);
    setPhase(0);
    setError("");
    setNotice("");
    setRunInput(null);
  }
  function changePrompt(value: string) {
    setPrompt(value);
    clearRun();
  }
  function input(overrides: Partial<GenerationInput> = {}): GenerationInput {
    return {
      prompt,
      maxTokens,
      temperature,
      topP,
      seed,
      cache,
      layer,
      head,
      trace: true,
      ...overrides,
    };
  }
  async function generate(
    kind: "step" | "run" | "inspect",
    overrides: Partial<GenerationInput> = {},
  ) {
    if (!ready || activeJob.current) return;
    activeJob.current = true;
    const controller = new AbortController();
    abort.current = controller;
    const payload = input(overrides);
    setBusy(kind);
    setError("");
    setNotice("");
    setRunInput(payload);
    try {
      if (kind === "run") {
        setSteps([]);
        setSelectedStep(0);
        setResult(null);
        setPhase(3);
        const finished = await stream<StepTrace, Generation>(
          "/api/generate/stream",
          payload,
          controller.signal,
          (token) => {
            setSteps((previous) => [...previous, token]);
            setSelectedStep(token.index);
          },
        );
        if (finished) {
          setResult(finished);
          setSteps(finished.steps);
          setSelectedStep(Math.max(0, finished.steps.length - 1));
        }
      } else {
        const next = await stream<StepTrace, Generation>(
          "/api/generate/stream",
          payload,
          controller.signal,
          () => {},
        );
        if (!next)
          throw new Error("The runtime ended without a generation result.");
        setResult(next);
        setSteps(next.steps);
        setSelectedStep(
          kind === "inspect"
            ? Math.min(selectedStep, next.steps.length - 1)
            : Math.max(0, next.steps.length - 1),
        );
        setPhase(kind === "inspect" ? 3 : 1);
      }
    } catch (failure) {
      if (isAbort(failure))
        setNotice(
          "Stopped. The tokens already received are still available to inspect.",
        );
      else setError(message(failure));
    } finally {
      activeJob.current = false;
      if (alive.current) setBusy(null);
    }
  }
  function step() {
    if (current && phase < 5) {
      setPhase((value) => value + 1);
      return;
    }
    const count = Math.min(64, steps.length + 1);
    void generate("step", { maxTokens: count });
  }
  function inspect(nextLayer: number, nextHead: number) {
    setLayer(nextLayer);
    setHead(nextHead);
    if (steps.length && !busy)
      void generate("inspect", {
        layer: nextLayer,
        head: nextHead,
        maxTokens: Math.max(1, steps.length),
        ...(runInput
          ? {
              temperature: runInput.temperature,
              topP: runInput.topP,
              seed: runInput.seed,
              cache: runInput.cache,
            }
          : {}),
      });
  }
  const actionLabel = !current
    ? "Step through"
    : phase < 5
      ? `Inspect ${stages[phase + 1].toLowerCase()}`
      : "Generate next token";

  return (
    <>
      <header className="masthead">
        <a
          className="brand"
          href="#walkthrough"
          onClick={(event) => {
            event.preventDefault();
            setView("walkthrough");
          }}
          aria-label="TensorTrace home"
        >
          <GraphIcon size={29} weight="fill" />
          <span>TensorTrace</span>
        </a>
        <nav aria-label="Workspace">
          <button
            className={view === "walkthrough" ? "nav-link active" : "nav-link"}
            aria-current={view === "walkthrough" ? "page" : undefined}
            onClick={() => setView("walkthrough")}
          >
            Walkthrough
          </button>
          <button
            className={view === "compare" ? "nav-link active" : "nav-link"}
            aria-current={view === "compare" ? "page" : undefined}
            onClick={() => setView("compare")}
          >
            Compare
          </button>
        </nav>
        <div className="header-end">
          <span className="engine-label">
            <span className={`status-dot ${ready ? "online" : ""}`} />
            {loading ? "Connecting" : ready ? "Rust · CPU" : "Runtime offline"}
          </span>
          <button
            ref={aboutTrigger}
            className="icon-button"
            aria-label="About this engine"
            onClick={() => setAbout(true)}
          >
            <InfoIcon size={21} />
          </button>
        </div>
      </header>

      <main className="workspace">
        {!ready && (
          <div
            className={`runtime-notice ${loading ? "connecting" : ""}`}
            role="status"
          >
            {loading ? (
              <CircleNotchIcon size={21} className="spin" />
            ) : (
              <WarningCircleIcon size={23} />
            )}
            <div>
              <strong>
                {loading
                  ? "Connecting to the Rust runtime"
                  : health?.status === "missing_model"
                    ? "Download the model to begin"
                    : "The runtime is not available"}
              </strong>
              <p>
                {loading
                  ? "Model metadata and traces will come from your local engine."
                  : health?.error ||
                    connectionError ||
                    "Run the setup and start commands from the repository README, then reconnect."}
              </p>
              {!loading && <code>make setup && make serve</code>}
            </div>
            {!loading && (
              <button className="text-button" onClick={() => void connect()}>
                <ArrowClockwiseIcon size={16} />
                Reconnect
              </button>
            )}
          </div>
        )}

        {view === "walkthrough" ? (
          <>
            <section className="intro">
              <h1>One token, from start to finish.</h1>
              <p>Inspect the calculations behind the next piece of text.</p>
            </section>
            <form
              className="prompt-form"
              onSubmit={(event) => {
                event.preventDefault();
                if (!busy) step();
              }}
            >
              <label className="prompt-input">
                <span>Prompt</span>
                <input
                  value={prompt}
                  onChange={(event) => changePrompt(event.target.value)}
                  disabled={Boolean(busy)}
                  maxLength={4096}
                  placeholder="Start a short story…"
                  autoComplete="off"
                  spellCheck={false}
                />
              </label>
              <button
                className="primary-button"
                type="submit"
                disabled={
                  !ready ||
                  !prompt.trim() ||
                  Boolean(busy) ||
                  (phase === 5 && steps.length >= 64)
                }
              >
                {busy === "step" || busy === "inspect" ? (
                  <>
                    <CircleNotchIcon size={18} className="spin" />
                    Computing
                  </>
                ) : (
                  <>
                    {actionLabel}
                    <ArrowRightIcon size={16} />
                  </>
                )}
              </button>
              {busy ? (
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => abort.current?.abort()}
                >
                  <SquareIcon size={15} weight="fill" />
                  Stop
                </button>
              ) : (
                <button
                  className="secondary-button run-button"
                  type="button"
                  disabled={!ready || !prompt.trim()}
                  onClick={() => void generate("run")}
                >
                  <PlayIcon size={15} weight="fill" />
                  Run
                </button>
              )}
              <button
                className="text-button reset-button"
                type="button"
                onClick={clearRun}
                disabled={Boolean(busy) || !runDone}
              >
                <ArrowClockwiseIcon size={16} />
                Reset
              </button>
            </form>
            <div className="run-options">
              <span className="model-caption">
                {model ? (
                  <>
                    {model.name} <span>·</span>{" "}
                    {prettyNumber(model.parameterCount / 1e6, 1)}M parameters{" "}
                    <span>·</span> {model.config.seqLen} token context
                  </>
                ) : (
                  "Model details appear when the runtime is ready"
                )}
              </span>
              <button
                className={`settings-button ${settingsOpen ? "selected" : ""}`}
                aria-expanded={settingsOpen}
                onClick={() => setSettingsOpen((value) => !value)}
              >
                <SlidersHorizontalIcon size={16} />
                Generation settings
              </button>
            </div>
            {settingsOpen && (
              <div className="settings-panel">
                <label>
                  Run length
                  <select
                    value={maxTokens}
                    disabled={Boolean(busy)}
                    onChange={(event) =>
                      setMaxTokens(Number(event.target.value))
                    }
                  >
                    {[4, 8, 16, 32, 64].map((value) => (
                      <option key={value} value={value}>
                        {value} tokens
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Temperature
                  <input
                    type="number"
                    value={temperature}
                    disabled={Boolean(busy)}
                    min="0"
                    max="2"
                    step="0.1"
                    onChange={(event) => {
                      clearRun();
                      setTemperature(
                        Math.min(2, Math.max(0, Number(event.target.value))),
                      );
                    }}
                  />
                </label>
                <label>
                  Top-p
                  <input
                    type="number"
                    value={topP}
                    disabled={Boolean(busy)}
                    min="0.01"
                    max="1"
                    step="0.05"
                    onChange={(event) => {
                      clearRun();
                      setTopP(
                        Math.min(1, Math.max(0.01, Number(event.target.value))),
                      );
                    }}
                  />
                </label>
                <label>
                  Seed
                  <input
                    type="number"
                    value={seed}
                    disabled={Boolean(busy)}
                    min="0"
                    max="4294967295"
                    step="1"
                    onChange={(event) => {
                      clearRun();
                      setSeed(
                        Math.min(
                          4294967295,
                          Math.max(0, Math.floor(Number(event.target.value))),
                        ),
                      );
                    }}
                  />
                </label>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={cache}
                    disabled={Boolean(busy)}
                    onChange={(event) => {
                      clearRun();
                      setCache(event.target.checked);
                    }}
                  />
                  <span>KV cache enabled</span>
                </label>
                <p>
                  Temperature 0 uses greedy decoding. Step-through requests
                  replay the sequence with the same seed.
                </p>
              </div>
            )}
            {error && (
              <div className="inline-message error" role="alert">
                <WarningCircleIcon size={19} />
                <span>{error}</span>
                <button
                  className="icon-button"
                  aria-label="Dismiss error"
                  onClick={() => setError("")}
                >
                  <XIcon size={16} />
                </button>
              </div>
            )}
            {notice && (
              <div className="inline-message" role="status">
                <InfoIcon size={18} />
                <span>{notice}</span>
              </div>
            )}

            <section className="computation-path" aria-label="Inference stages">
              {stages.map((stage, index) => (
                <div className="path-segment" key={stage}>
                  <button
                    className={`path-step ${phase === index ? "selected" : ""}`}
                    aria-current={phase === index ? "step" : undefined}
                    onClick={() => setPhase(index)}
                  >
                    <span className="path-name">{stage}</span>
                    <span className="path-visual">
                      {index === 0 ? (
                        <span className="text-preview">
                          {prompt.length > 35
                            ? prompt.slice(0, 35) + "…"
                            : prompt || "Your text"}
                        </span>
                      ) : index === 1 ? (
                        <BracketsSquareIcon size={34} weight="duotone" />
                      ) : index === 2 ? (
                        <StackIcon size={34} weight="duotone" />
                      ) : index === 3 ? (
                        <GraphIcon size={34} weight="duotone" />
                      ) : index === 4 ? (
                        <ChartLineIcon size={34} weight="duotone" />
                      ) : (
                        <span
                          className={`next-preview ${current ? "has-token" : ""}`}
                        >
                          {current ? tokenText(current.token) : "—"}
                        </span>
                      )}
                    </span>
                    <span className="path-caption">
                      {index === 0
                        ? "UTF-8 text"
                        : index === 1
                          ? current
                            ? `${inputTokens.length} token IDs`
                            : "Byte-pair encoding"
                          : index === 2
                            ? model && current
                              ? `${inputTokens.length} × ${model.config.dim}`
                              : "Learned vectors"
                            : index === 3
                              ? model
                                ? `${model.config.nLayers} layers · attention + MLP`
                                : "Attention + MLP"
                              : index === 4
                                ? model
                                  ? `${prettyNumber(model.config.vocabSize, 0)} vocabulary scores`
                                  : "Vocabulary logits"
                                : runInput?.temperature
                                  ? "Temperature / top-p"
                                  : "Greedy / sampling"}
                    </span>
                  </button>
                  {index < stages.length - 1 && (
                    <ArrowRightIcon className="path-arrow" size={22} />
                  )}
                </div>
              ))}
            </section>

            <section className="inspection-grid" aria-busy={Boolean(busy)}>
              <div className="stage-detail">
                <div className="section-title">
                  <div>
                    <h2>
                      {phase === 0
                        ? "The starting point"
                        : phase === 1
                          ? "Text becomes token IDs"
                          : phase === 2
                            ? "Inside the embedding"
                            : phase === 3
                              ? "Inside the transformer"
                              : phase === 4
                                ? "From scores to probabilities"
                                : "One more piece of text"}
                    </h2>
                  </div>
                  {current && (
                    <span className="subtle-label">
                      Prediction {selectedStep + 1}
                    </span>
                  )}
                </div>
                {phase === 3 && (
                  <div className="matrix-controls">
                    <label className="sr-only" htmlFor="layer">
                      Attention layer
                    </label>
                    <select
                      id="layer"
                      value={layer}
                      disabled={!current || Boolean(busy)}
                      onChange={(event) =>
                        inspect(Number(event.target.value), head)
                      }
                    >
                      {Array.from(
                        { length: model?.config.nLayers ?? 1 },
                        (_, i) => (
                          <option value={i} key={i}>
                            Layer {i + 1}
                          </option>
                        ),
                      )}
                    </select>
                    <label className="sr-only" htmlFor="head">
                      Attention head
                    </label>
                    <select
                      id="head"
                      value={head}
                      disabled={!current || Boolean(busy)}
                      onChange={(event) =>
                        inspect(layer, Number(event.target.value))
                      }
                    >
                      {Array.from(
                        { length: model?.config.nHeads ?? 1 },
                        (_, i) => (
                          <option value={i} key={i}>
                            Head {i + 1}
                          </option>
                        ),
                      )}
                    </select>
                    {busy === "inspect" && (
                      <span className="subtle-label">
                        <CircleNotchIcon className="spin" size={14} />
                        Capturing selection
                      </span>
                    )}
                  </div>
                )}
                {!current ? (
                  <div className="stage-empty">
                    <GraphIcon size={46} weight="duotone" />
                    <h3>Your first token starts here.</h3>
                    <p>
                      Step through a prompt to inspect the actual vectors,
                      attention weights, and predictions from the model.
                    </p>
                    <span>
                      No sample values. Every trace comes from the Rust engine.
                    </span>
                  </div>
                ) : (
                  <>
                    {phase === 0 && (
                      <div className="text-inspection">
                        <span className="small-label">CURRENT CONTEXT</span>
                        <p>
                          {prompt}
                          {selectedStep > 0
                            ? (steps[selectedStep - 1]?.text ?? "")
                            : ""}
                        </p>
                        <div className="text-facts">
                          <span>
                            <strong>
                              {new TextEncoder().encode(prompt).length}
                            </strong>
                            prompt bytes
                          </span>
                          <span>
                            <strong>{inputTokens.length}</strong>context tokens
                          </span>
                          <span>
                            <strong>{model?.config.seqLen}</strong>context
                            capacity
                          </span>
                        </div>
                      </div>
                    )}
                    {phase === 1 && (
                      <div className="token-inspection">
                        {inputTokens.map((token, index) => (
                          <div
                            className={`token-detail ${index === inputTokens.length - 1 ? "current" : ""}`}
                            key={`${index}-${token.id}`}
                            title={`Position ${index}: ${JSON.stringify(token.text)}`}
                          >
                            <span className="token-position">{index}</span>
                            <strong>{tokenText(token)}</strong>
                            <span>{token.id}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {phase === 2 && (
                      <>
                        <VectorChart values={current.embedding} />
                        <p className="detail-caption">
                          First {current.embedding.length} of{" "}
                          {model?.config.dim} embedding values for the final
                          input token,{" "}
                          <code>
                            {tokenText(inputTokens[inputTokens.length - 1])}
                          </code>
                          .
                        </p>
                      </>
                    )}
                    {phase === 3 && (
                      <>
                        <AttentionChart
                          matrix={current.attention}
                          pieces={inputTokens.map((token) => tokenText(token))}
                        />
                        <p className="detail-caption">
                          Captured layer {current.layer + 1}, head{" "}
                          {current.head + 1}. Query rows attend to key columns.
                          Future positions are masked. The outlined row is the
                          final input position.
                        </p>
                        <p className="formula">softmax(QKᵀ / √dₖ + mask)V</p>
                      </>
                    )}
                    {phase === 4 && (
                      <>
                        <div className="logit-table">
                          <div className="logit-row logit-head">
                            <span>Token</span>
                            <span>Raw logit</span>
                            <span>Probability</span>
                          </div>
                          {current.topTokens.slice(0, 7).map((token) => (
                            <div className="logit-row" key={token.id}>
                              <code>{tokenText(token)}</code>
                              <span>{token.logit.toFixed(4)}</span>
                              <span>{percent(token.probability)}</span>
                            </div>
                          ))}
                        </div>
                        <p className="formula">
                          pᵢ = exp(logitᵢ − max) / Σ exp(logitⱼ − max)
                        </p>
                      </>
                    )}
                    {phase === 5 && (
                      <div className="selected-explanation">
                        <div className="large-token">
                          {tokenText(current.token)}
                        </div>
                        <span>
                          Token ID <code>{current.token.id}</code> ·{" "}
                          {percent(current.selectedProbability)} raw model
                          probability
                        </span>
                        <p>
                          {runInput?.temperature === 0
                            ? "Greedy decoding selects the highest-scoring token. It is appended to the context before the next prediction."
                            : `Seeded sampling uses temperature ${runInput?.temperature} and top-p ${runInput?.topP}. The chosen token is appended to the context before the next prediction.`}
                        </p>
                      </div>
                    )}
                    <p className="stage-explanation">
                      {stageDescriptions[phase]}
                    </p>
                    <details className="activation-inspector">
                      <summary>
                        Inspect captured vectors and activations{" "}
                        <ArrowRightIcon size={13} />
                      </summary>
                      <p>
                        These are the first {current.embedding.length}{" "}
                        components of the captured vectors; complete model
                        dimensions are shown above.
                      </p>
                      {[
                        ["Embedding", current.embedding],
                        ["Query (after RoPE)", current.query],
                        ["Key (after RoPE)", current.key],
                        ["Value", current.value],
                        ["Attention output", current.attentionOutput],
                        ["Feed-forward output", current.feedForward],
                        ["Final hidden state", current.hiddenState],
                      ].map(([label, values]) => (
                        <div className="activation-row" key={label as string}>
                          <span>{label as string}</span>
                          <code>
                            {(values as number[])
                              .map((value) => value.toFixed(4))
                              .join(", ")}
                          </code>
                        </div>
                      ))}
                    </details>
                  </>
                )}
              </div>
              <aside className="predictions">
                <h2>Possible next tokens</h2>
                {current ? (
                  <>
                    <Probabilities
                      candidates={current.topTokens}
                      selectedId={current.token.id}
                    />
                    <p className="probability-note">
                      Raw model probabilities, before temperature and top-p
                      filtering.
                    </p>
                    <div className="chosen-token">
                      <span>Next token:</span>
                      <strong>{tokenText(current.token)}</strong>
                    </div>
                    <p className="detail-caption">
                      One sampling step adds one token to the sequence.
                    </p>
                    <div className="prediction-footnotes">
                      <span>
                        Entropy
                        <strong>{current.entropy.toFixed(3)} nats</strong>
                      </span>
                      <span>
                        Forward pass
                        <strong>
                          {prettyNumber(current.timing.forwardMs, 2)} ms
                        </strong>
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="prediction-empty">
                    <ChartLineIcon size={32} />
                    <p>The model’s next-token distribution will appear here.</p>
                    <span>No inference has run yet.</span>
                  </div>
                )}
              </aside>
            </section>

            <section className="output-section" aria-label="Generated text">
              <div className="output-main">
                <div className="output-heading">
                  <span className="small-label">Generated text</span>
                  {steps.length > 0 && (
                    <span className="subtle-label">
                      {steps.length} {steps.length === 1 ? "token" : "tokens"}
                      {busy === "run" ? " · generating" : ""}
                    </span>
                  )}
                </div>
                <div className="output-text" aria-live="polite">
                  <span className="original-prompt">
                    {prompt || "Your prompt"}
                  </span>
                  {steps.length ? (
                    <span className="decoded-output">
                      {result?.generatedText ?? steps.at(-1)?.text ?? ""}
                    </span>
                  ) : (
                    <span className="output-placeholder"> …</span>
                  )}
                  {busy === "run" && <span className="typing-indicator" />}
                </div>
                {steps.length > 0 && (
                  <div
                    className="generated-token-picker"
                    role="group"
                    aria-label="Inspect generated tokens"
                  >
                    <span className="small-label">Inspect tokens</span>
                    {steps.map((token, index) => (
                      <button
                        key={`${index}-${token.token.id}`}
                        className={`output-token ${index === selectedStep ? "selected" : ""}`}
                        title={`Raw vocabulary piece, ID ${token.token.id}`}
                        aria-label={`Inspect generated token ${index + 1}: ${JSON.stringify(token.token.text)}`}
                        onClick={() => {
                          setSelectedStep(index);
                          setPhase(3);
                        }}
                      >
                        {tokenText(token.token)}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="output-status">
                <span>
                  <CpuIcon size={16} />
                  CPU runtime
                </span>
                <span>
                  <span
                    className={`status-dot ${(result?.stats.cacheEnabled ?? runInput?.cache ?? cache) ? "online" : ""}`}
                  />
                  KV cache{" "}
                  {(result?.stats.cacheEnabled ?? runInput?.cache ?? cache)
                    ? "enabled"
                    : "disabled"}
                </span>
                {result && (
                  <span>{memory(result.stats.kvCacheBytes)} reusable K/V</span>
                )}
              </div>
            </section>
            <footer className="workspace-footer">
              <span>
                Walkthroughs inspect a captured forward pass; tracing adds
                overhead.
              </span>
              <span>
                {result
                  ? `${prettyNumber(result.stats.totalMs, 1)} ms total · ${result.stats.stoppedReason.replaceAll("_", " ")}`
                  : "Use Compare for timings with tracing disabled."}
              </span>
            </footer>
          </>
        ) : (
          <Compare ready={ready} model={model} defaultPrompt={prompt} />
        )}
      </main>
      {about && (
        <div
          className="modal-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setAbout(false);
          }}
        >
          <section
            className="about-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="about-title"
          >
            <button
              className="icon-button close-dialog"
              aria-label="Close model details"
              onClick={() => setAbout(false)}
              autoFocus
            >
              <XIcon size={21} />
            </button>
            <GraphIcon className="about-icon" size={34} weight="duotone" />
            <span className="eyebrow">THE ENGINE BEHIND THE VIEW</span>
            <h2 id="about-title">
              Small enough to inspect.
              <br />
              Real enough to understand.
            </h2>
            <p>
              TensorTrace runs a pretrained language model through its own Rust
              CPU inference code. There is no remote inference API. The
              walkthrough exposes actual checkpoint values and forward-pass
              calculations.
            </p>
            {model && (
              <dl className="model-facts">
                <div>
                  <dt>Checkpoint</dt>
                  <dd>{model.name}</dd>
                </div>
                <div>
                  <dt>Architecture</dt>
                  <dd>{model.architecture}</dd>
                </div>
                <div>
                  <dt>Layers / attention heads</dt>
                  <dd>
                    {model.config.nLayers} / {model.config.nHeads}
                  </dd>
                </div>
                <div>
                  <dt>Embedding / hidden dimension</dt>
                  <dd>
                    {model.config.dim} / {model.config.hiddenDim}
                  </dd>
                </div>
                <div>
                  <dt>Parameters / weight bytes</dt>
                  <dd>
                    {prettyNumber(model.parameterCount, 0)} /{" "}
                    {memory(model.modelBytes)}
                  </dd>
                </div>
                <div>
                  <dt>Vocabulary / context</dt>
                  <dd>
                    {prettyNumber(model.config.vocabSize, 0)} /{" "}
                    {model.config.seqLen} tokens
                  </dd>
                </div>
              </dl>
            )}
            <p className="about-note">
              Attention weights describe information mixing; they are not an
              explanation of a model’s reasoning. Local benchmark results
              describe this machine and workload.
            </p>
            <a
              className="text-link"
              href="https://github.com/rohansikder/tensor-trace"
              target="_blank"
              rel="noreferrer"
            >
              Read the implementation <ArrowUpRightIcon size={16} />
            </a>
          </section>
        </div>
      )}
    </>
  );
}

function percent(value: number) {
  const n = value * 100;
  return `${n >= 1 ? n.toFixed(1) : n >= 0.01 ? n.toFixed(2) : n.toFixed(4)}%`;
}
function Probabilities({
  candidates,
  selectedId,
}: {
  candidates: Candidate[];
  selectedId: number;
}) {
  const shown = candidates.slice(0, 5);
  const other = Math.max(
    0,
    1 - shown.reduce((sum, token) => sum + token.probability, 0),
  );
  return (
    <div className="probability-list">
      {shown.map((token) => (
        <div
          className={`probability-row ${token.id === selectedId ? "selected" : ""}`}
          key={token.id}
        >
          <code title={`Token ${token.id}: ${JSON.stringify(token.text)}`}>
            {tokenText(token)}
          </code>
          <span className="probability-track">
            <span
              style={{
                width: `${Math.max(0, Math.min(100, token.probability * 100))}%`,
              }}
            />
          </span>
          <span className="probability-value">
            {percent(token.probability)}
          </span>
        </div>
      ))}
      <div className="probability-row other">
        <span>Other</span>
        <span className="probability-track">
          <span style={{ width: `${other * 100}%` }} />
        </span>
        <span className="probability-value">{percent(other)}</span>
      </div>
    </div>
  );
}
