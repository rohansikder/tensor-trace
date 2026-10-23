export interface Model {
  name: string;
  architecture: string;
  config: {
    dim: number;
    hiddenDim: number;
    nLayers: number;
    nHeads: number;
    nKvHeads: number;
    vocabSize: number;
    seqLen: number;
    sharedClassifier: boolean;
  };
  parameterCount: number;
  modelBytes: number;
  tokenizerBytes: number;
  engine: string;
}
export interface Health {
  status: "ready" | "missing_model";
  engine: string;
  error?: string;
}
export interface Token {
  id: number;
  text: string;
}
export interface Candidate extends Token {
  probability: number;
  logit: number;
}
export interface StepTrace {
  index: number;
  position: number;
  token: Token;
  text: string;
  contextTokens: Token[];
  attention: number[][];
  embedding: number[];
  hiddenState: number[];
  query: number[];
  key: number[];
  value: number[];
  attentionOutput: number[];
  feedForward: number[];
  topTokens: Candidate[];
  entropy: number;
  selectedProbability: number;
  layer: number;
  head: number;
  timing: { forwardMs: number; samplingMs: number; totalMs: number };
}
export interface Stats {
  promptTokens: number;
  generatedTokens: number;
  decodeTokens: number;
  prefillMs: number;
  decodeMs: number;
  samplingMs: number;
  totalMs: number;
  tokensPerSecond: number;
  decodeMsPerToken: number;
  kvCacheBytes: number;
  cacheEnabled: boolean;
  contextCapacity: number;
  stoppedReason: string;
  seed: number;
}
export interface Generation {
  text: string;
  generatedText: string;
  promptTokens: Token[];
  generatedTokens: Token[];
  steps: StepTrace[];
  stats: Stats;
}
export interface GenerationInput {
  prompt: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  seed: number;
  cache: boolean;
  layer: number;
  head: number;
  trace: boolean;
}
