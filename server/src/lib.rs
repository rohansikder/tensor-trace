//! The local HTTP boundary and repeatable CPU benchmark protocol.
use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, Json, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{any, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tensor_trace_engine::{Engine, GenerationRequest, GenerationResult};
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tower_http::services::{ServeDir, ServeFile};

pub const MAX_PROMPT_BYTES: usize = 4096;
pub const MAX_GENERATED_TOKENS: usize = 64;
const RUN_DEADLINE: Duration = Duration::from_secs(120);

#[derive(Clone)]
pub struct AppState {
    pub engine: Option<Arc<Engine>>,
    pub load_error: Option<String>,
    pub gate: Arc<Semaphore>,
}
impl AppState {
    pub fn load(
        model: &std::path::Path,
        tokenizer: &std::path::Path,
        concurrent_jobs: usize,
    ) -> Self {
        match Engine::load(model, tokenizer) {
            Ok(engine) => Self::ready(engine, concurrent_jobs),
            Err(error) => Self {
                engine: None,
                load_error: Some(format!(
                    "{error}. Run python3 scripts/fetch_model.py, then restart the server."
                )),
                gate: Arc::new(Semaphore::new(concurrent_jobs.clamp(1, 2))),
            },
        }
    }
    pub fn ready(engine: Engine, concurrent_jobs: usize) -> Self {
        Self {
            engine: Some(Arc::new(engine)),
            load_error: None,
            gate: Arc::new(Semaphore::new(concurrent_jobs.clamp(1, 2))),
        }
    }
    fn require_engine(&self) -> Result<Arc<Engine>, ApiError> {
        self.engine.clone().ok_or_else(|| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "missing_model",
                self.load_error
                    .clone()
                    .unwrap_or_else(|| "Model unavailable".into()),
            )
        })
    }
    fn permit(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        self.gate.clone().try_acquire_owned().map_err(|_| {
            ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "runtime_busy",
                "Another inference or benchmark is running. Wait for it to finish or cancel it.",
            )
        })
    }
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}
impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }
    fn runtime(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "runtime_error", message)
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({"error":self.message,"code":self.code})),
        )
            .into_response()
    }
}

pub fn router(state: AppState, web_dir: Option<PathBuf>) -> Router {
    let api = Router::new()
        .route("/", any(api_not_found))
        .route("/health", get(health))
        .route("/model", get(model))
        .route("/tokenize", post(tokenize))
        .route("/generate", post(generate))
        .route("/generate/stream", post(generate_stream))
        .route("/benchmark", post(benchmark))
        .route("/benchmark/stream", post(benchmark_stream))
        .fallback(api_not_found)
        .layer(DefaultBodyLimit::max(16 * 1024))
        .with_state(state);
    let outer = Router::new()
        .route("/api/", any(api_not_found))
        .nest("/api", api);
    if let Some(directory) = web_dir {
        let index = directory.join("index.html");
        outer.fallback_service(ServeDir::new(directory).not_found_service(ServeFile::new(index)))
    } else {
        outer.fallback(api_not_found)
    }
}
async fn api_not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "not_found", "Unknown route")
}
async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(
        json!({"status": if state.engine.is_some() {"ready"} else {"missing_model"}, "engine":"tensor-trace-rust", "model":state.engine.as_ref().map(|e|e.metadata()), "error":state.load_error}),
    )
}
async fn model(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    Ok(Json(json!(state.require_engine()?.metadata())))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizeRequest {
    text: String,
}
fn validate_text(text: &str) -> Result<(), ApiError> {
    if text.trim().is_empty() {
        return Err(ApiError::invalid("Enter a nonempty prompt"));
    }
    if text.len() > MAX_PROMPT_BYTES {
        return Err(ApiError::invalid(format!(
            "Prompt exceeds {MAX_PROMPT_BYTES} UTF-8 bytes"
        )));
    }
    Ok(())
}
fn request_json<T>(request: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    request
        .map(|Json(value)| value)
        .map_err(|error| ApiError::new(error.status(), "invalid_json", error.body_text()))
}

async fn tokenize(
    State(state): State<AppState>,
    request: Result<Json<TokenizeRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let request = request_json(request)?;
    validate_text(&request.text)?;
    let engine = state.require_engine()?;
    let _permit = state.permit()?;
    let tokens = engine.tokenize(&request.text).map_err(ApiError::invalid)?;
    Ok(Json(json!(tokens)))
}

pub fn validate_generation(engine: &Engine, request: &GenerationRequest) -> Result<(), ApiError> {
    validate_text(&request.prompt)?;
    if !(1..=MAX_GENERATED_TOKENS).contains(&request.max_tokens) {
        return Err(ApiError::invalid("maxTokens must be between 1 and 64"));
    }
    if !request.temperature.is_finite() || !(0.0..=2.0).contains(&request.temperature) {
        return Err(ApiError::invalid(
            "temperature must be a finite number from 0 to 2",
        ));
    }
    if !request.top_p.is_finite() || request.top_p <= 0.0 || request.top_p > 1.0 {
        return Err(ApiError::invalid("topP must be a finite number in (0, 1]"));
    }
    let metadata = engine.metadata();
    if request.layer >= metadata.config.n_layers || request.head >= metadata.config.n_heads {
        return Err(ApiError::invalid(
            "Selected layer or head is outside this model",
        ));
    }
    let token_count = engine
        .tokenize(&request.prompt)
        .map_err(ApiError::invalid)?
        .ids
        .len();
    if token_count + request.max_tokens > metadata.config.seq_len {
        return Err(ApiError::invalid(format!("Prompt ({token_count} tokens) plus output budget ({}) exceeds model context capacity ({})", request.max_tokens, metadata.config.seq_len)));
    }
    Ok(())
}

async fn generate(
    State(state): State<AppState>,
    request: Result<Json<GenerationRequest>, JsonRejection>,
) -> Result<Json<GenerationResult>, ApiError> {
    let request = request_json(request)?;
    let engine = state.require_engine()?;
    let permit = state.permit()?;
    validate_generation(&engine, &request)?;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let deadline = Instant::now() + RUN_DEADLINE;
        let mut timed_out = false;
        let result = engine.generate_stream(&request, |_| {
            timed_out = Instant::now() >= deadline;
            !timed_out
        });
        if timed_out {
            return Err("Generation exceeded its 120-second deadline".into());
        }
        result
    })
    .await
    .map_err(|error| ApiError::runtime(error.to_string()))?
    .map_err(ApiError::runtime)?;
    Ok(Json(result))
}

// Never hold the CPU slot forever for a client that keeps its TCP connection open
// but stops consuming SSE. Channel backpressure participates in the same deadline.
fn send_event(
    sender: &mpsc::Sender<(&'static str, String)>,
    mut event: (&'static str, String),
    deadline: Instant,
) -> bool {
    loop {
        match sender.try_send(event) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(returned)) => {
                event = returned;
                if Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

async fn generate_stream(
    State(state): State<AppState>,
    request: Result<Json<GenerationRequest>, JsonRejection>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let request = request_json(request)?;
    let engine = state.require_engine()?;
    let permit = state.permit()?;
    validate_generation(&engine, &request)?;
    let (sender, receiver) = mpsc::channel::<(&'static str, String)>(4);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let deadline = Instant::now() + RUN_DEADLINE;
        let mut timed_out = false;
        let result = engine.generate_stream(&request, |step| {
            if Instant::now() >= deadline {
                timed_out = true;
                return false;
            }
            if sender.is_closed() {
                return false;
            }
            match serde_json::to_string(step) {
                Ok(data) => {
                    let sent = send_event(&sender, ("token", data), deadline);
                    timed_out = !sent && Instant::now() >= deadline;
                    sent
                }
                Err(_) => false,
            }
        });
        if sender.is_closed() {
            return;
        }
        let event = if timed_out {
            ("error", json!({"error":"Generation exceeded its 120-second deadline","code":"deadline_exceeded"}).to_string())
        } else {
            match result {
                Ok(result) => match serde_json::to_string(&result) {
                    Ok(data) => ("done", data),
                    Err(error) => (
                        "error",
                        json!({"error":error.to_string(),"code":"serialization_error"}).to_string(),
                    ),
                },
                Err(error) => (
                    "error",
                    json!({"error":error,"code":"runtime_error"}).to_string(),
                ),
            }
        };
        let _ = send_event(&sender, event, deadline);
    });
    let stream = ReceiverStream::new(receiver)
        .map(|(name, data)| Ok(Event::default().event(name).data(data)));
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(10))
            .text("runtime active"),
    ))
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct BenchmarkRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub context_lengths: Vec<usize>,
    pub repeats: usize,
}
impl Default for BenchmarkRequest {
    fn default() -> Self {
        Self {
            prompt: "Once upon a time".into(),
            max_tokens: 8,
            context_lengths: vec![8, 16, 32],
            repeats: 3,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkMeasurement {
    pub prefill_ms: f64,
    pub time_to_first_token_ms: f64,
    pub forward_decode_ms: f64,
    pub decode_ms_per_token: f64,
    pub tokens_per_second: f64,
    pub total_ms: f64,
    pub kv_cache_bytes: usize,
    pub generated_tokens: usize,
    pub decode_tokens: usize,
    pub samples_ms_per_token: Vec<f64>,
}
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkPoint {
    pub context_length: usize,
    pub cache_on: BenchmarkMeasurement,
    pub cache_off: BenchmarkMeasurement,
    pub tokens_match: bool,
    pub speedup: f64,
}
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub model: String,
    pub points: Vec<BenchmarkPoint>,
    pub repeats: usize,
    pub max_tokens: usize,
    pub elapsed_ms: f64,
    pub method: String,
    pub tracing: bool,
    pub prompt_construction: String,
}

fn validate_benchmark(engine: &Engine, request: &BenchmarkRequest) -> Result<(), String> {
    validate_text(&request.prompt).map_err(|error| error.message)?;
    if !(2..=16).contains(&request.max_tokens) {
        return Err("Benchmark maxTokens must be between 2 and 16 to measure steady decode".into());
    }
    if !(1..=3).contains(&request.repeats) {
        return Err("Benchmark repeats must be between 1 and 3".into());
    }
    if request.context_lengths.is_empty() || request.context_lengths.len() > 4 {
        return Err("Provide between 1 and 4 context lengths".into());
    }
    let capacity = engine.metadata().config.seq_len;
    let mut seen = std::collections::HashSet::new();
    for &length in &request.context_lengths {
        if !(2..=128).contains(&length) || length + request.max_tokens > capacity {
            return Err(format!("Context lengths must be 2..128 and leave room for output within the {capacity}-token model capacity"));
        }
        if !seen.insert(length) {
            return Err("Context lengths must be unique".into());
        }
    }
    let work: usize = request
        .context_lengths
        .iter()
        .map(|length| (length + request.max_tokens) * request.max_tokens * (request.repeats + 1))
        .sum();
    if work > 12_000 {
        return Err(
            "Benchmark workload is too large; reduce context lengths, output tokens or repeats"
                .into(),
        );
    }
    Ok(())
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}
fn measurements(results: &[GenerationResult]) -> BenchmarkMeasurement {
    let first = &results[0].stats;
    BenchmarkMeasurement {
        prefill_ms: median(results.iter().map(|run| run.stats.prefill_ms).collect()),
        time_to_first_token_ms: median(
            results
                .iter()
                .map(|run| run.stats.time_to_first_token_ms)
                .collect(),
        ),
        forward_decode_ms: median(
            results
                .iter()
                .map(|run| run.stats.forward_decode_ms)
                .collect(),
        ),
        decode_ms_per_token: median(
            results
                .iter()
                .map(|run| run.stats.decode_ms_per_token)
                .collect(),
        ),
        tokens_per_second: median(
            results
                .iter()
                .map(|run| run.stats.tokens_per_second)
                .collect(),
        ),
        total_ms: median(results.iter().map(|run| run.stats.total_ms).collect()),
        kv_cache_bytes: first.kv_cache_bytes,
        generated_tokens: first.generated_tokens,
        decode_tokens: first.decode_tokens,
        samples_ms_per_token: results
            .iter()
            .map(|run| run.stats.decode_ms_per_token)
            .collect(),
    }
}

pub fn run_benchmark(
    engine: &Engine,
    request: &BenchmarkRequest,
) -> Result<BenchmarkResult, String> {
    run_benchmark_cancellable(engine, request, || false)
}

fn run_benchmark_cancellable(
    engine: &Engine,
    request: &BenchmarkRequest,
    cancelled: impl Fn() -> bool,
) -> Result<BenchmarkResult, String> {
    validate_benchmark(engine, request)?;
    let started = Instant::now();
    let deadline = started + RUN_DEADLINE;
    let tokenization = engine.tokenize(&request.prompt)?;
    // Keep one BOS token; repeat only the seed's content pieces.
    let content = if tokenization.ids.len() > 1 {
        &tokenization.ids[1..]
    } else {
        &tokenization.ids[..]
    };
    if content.is_empty() {
        return Err("Benchmark seed must contain at least one text token".into());
    }
    let mut points = Vec::new();
    for &context_length in &request.context_lengths {
        let mut ids = vec![tokenization.ids[0]];
        ids.extend(content.iter().copied().cycle().take(context_length - 1));
        let mut config = GenerationRequest {
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
            cache: true,
            layer: 0,
            head: 0,
            trace: false,
            stop_on_eos: false,
        };
        let mut run = |cache: bool| -> Result<GenerationResult, String> {
            if cancelled() {
                return Err("Benchmark cancelled".into());
            }
            if Instant::now() >= deadline {
                return Err("Benchmark exceeded its 120-second deadline".into());
            }
            config.cache = cache;
            let mut timed_out = false;
            let mut was_cancelled = false;
            let result = engine.generate_tokens_stream(&ids, &config, |_| {
                timed_out = Instant::now() >= deadline;
                was_cancelled = cancelled();
                !timed_out && !was_cancelled
            })?;
            if was_cancelled {
                return Err("Benchmark cancelled".into());
            }
            if timed_out {
                return Err("Benchmark exceeded its 120-second deadline".into());
            }
            if result.generated_tokens.len() != request.max_tokens {
                return Err("Engine did not generate the complete benchmark token budget".into());
            }
            Ok(result)
        };
        let warm_on = run(true)?;
        let warm_off = run(false)?;
        let warm_ids: Vec<_> = warm_on
            .generated_tokens
            .iter()
            .map(|token| token.id)
            .collect();
        if warm_ids
            != warm_off
                .generated_tokens
                .iter()
                .map(|token| token.id)
                .collect::<Vec<_>>()
        {
            return Err("Cached and uncached output tokens differ during warmup".into());
        }
        let (mut on, mut off) = (Vec::new(), Vec::new());
        for repeat in 0..request.repeats {
            let (cached, uncached) = if repeat % 2 == 0 {
                (run(true)?, run(false)?)
            } else {
                let uncached = run(false)?;
                (run(true)?, uncached)
            };
            if cached
                .generated_tokens
                .iter()
                .map(|token| token.id)
                .collect::<Vec<_>>()
                != warm_ids
                || uncached
                    .generated_tokens
                    .iter()
                    .map(|token| token.id)
                    .collect::<Vec<_>>()
                    != warm_ids
            {
                return Err("Greedy token sequence changed between benchmark runs".into());
            }
            on.push(cached);
            off.push(uncached);
        }
        let cache_on = measurements(&on);
        let cache_off = measurements(&off);
        let speedup =
            cache_off.decode_ms_per_token / cache_on.decode_ms_per_token.max(f64::MIN_POSITIVE);
        points.push(BenchmarkPoint {
            context_length,
            cache_on,
            cache_off,
            tokens_match: true,
            speedup,
        });
    }
    Ok(BenchmarkResult { model:engine.metadata().name,points,repeats:request.repeats,max_tokens:request.max_tokens,elapsed_ms:started.elapsed().as_secs_f64()*1000.0,method:"Median steady-decode time per token after one warmup per mode; alternating mode order; identical greedy tokens; first-token prefill measured separately".into(),tracing:false,prompt_construction:"One BOS token followed by repeated token IDs from the seed prompt, truncated to each exact context length".into() })
}
async fn benchmark(
    State(state): State<AppState>,
    request: Result<Json<BenchmarkRequest>, JsonRejection>,
) -> Result<Json<BenchmarkResult>, ApiError> {
    let request = request_json(request)?;
    let engine = state.require_engine()?;
    validate_benchmark(&engine, &request).map_err(ApiError::invalid)?;
    let permit = state.permit()?;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        run_benchmark(&engine, &request)
    })
    .await
    .map_err(|error| ApiError::runtime(error.to_string()))?
    .map_err(ApiError::runtime)?;
    Ok(Json(result))
}

// Return the SSE body before starting CPU work, so disconnecting the browser
// drops its receiver and cancellation can be observed at the next token boundary.
async fn benchmark_stream(
    State(state): State<AppState>,
    request: Result<Json<BenchmarkRequest>, JsonRejection>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let request = request_json(request)?;
    let engine = state.require_engine()?;
    validate_benchmark(&engine, &request).map_err(ApiError::invalid)?;
    let permit = state.permit()?;
    let (sender, receiver) = mpsc::channel::<(&'static str, String)>(1);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let result = run_benchmark_cancellable(&engine, &request, || sender.is_closed());
        if sender.is_closed() {
            return;
        }
        let event = match result {
            Ok(result) => match serde_json::to_string(&result) {
                Ok(data) => ("done", data),
                Err(error) => (
                    "error",
                    json!({"error":error.to_string(),"code":"serialization_error"}).to_string(),
                ),
            },
            Err(error) => (
                "error",
                json!({"error":error,"code":"runtime_error"}).to_string(),
            ),
        };
        let _ = sender.try_send(event);
    });
    let stream = ReceiverStream::new(receiver)
        .map(|(name, data)| Ok(Event::default().event(name).data(data)));
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("benchmark active"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    fn missing() -> AppState {
        AppState {
            engine: None,
            load_error: Some("Run python3 scripts/fetch_model.py".into()),
            gate: Arc::new(Semaphore::new(1)),
        }
    }
    #[tokio::test]
    async fn missing_model_is_actionable_and_health_is_available() {
        let app = router(missing(), None);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let data: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(data["status"], "missing_model");
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/model")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 503);
    }
    #[tokio::test]
    async fn unknown_api_route_returns_json_not_spa_html() {
        let app = router(missing(), Some(PathBuf::from("../web/dist")));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/does-not-exist")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 404);
        assert_eq!(response.headers()["content-type"], "application/json");
    }
    #[test]
    fn non_consuming_stream_cannot_hold_the_execution_gate_forever() {
        let (sender, _receiver) = mpsc::channel(1);
        sender.try_send(("token", "first".into())).unwrap();
        let started = Instant::now();
        assert!(!send_event(
            &sender,
            ("token", "second".into()),
            started + Duration::from_millis(15)
        ));
        assert!(started.elapsed() < Duration::from_millis(200));
    }
    #[test]
    fn work_gate_and_text_limit_are_bounded() {
        let state = missing();
        let permit = state.permit().unwrap();
        assert!(state.permit().is_err());
        drop(permit);
        assert!(state.permit().is_ok());
        assert!(validate_text("").is_err());
        assert!(validate_text(&"x".repeat(MAX_PROMPT_BYTES + 1)).is_err());
        assert!(validate_text("hello").is_ok());
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 2.0]), 3.0);
    }
    fn fixture() -> AppState {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../validation/fixtures");
        AppState::ready(
            Engine::load(root.join("tiny-shared.bin"), root.join("tokenizer.bin")).unwrap(),
            1,
        )
    }
    fn generation_body() -> axum::body::Body {
        axum::body::Body::from(
            json!({"prompt":"abc","maxTokens":3,"temperature":0,"trace":true,"stopOnEos":false})
                .to_string(),
        )
    }
    #[tokio::test]
    async fn real_fixture_generation_and_sse_return_matching_computed_tokens() {
        let app = router(fixture(), None);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/generate")
                    .header("content-type", "application/json")
                    .body(generation_body())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let value: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(value["stats"]["generatedTokens"], 3);
        assert_eq!(value["steps"].as_array().unwrap().len(), 3);
        assert!(value["steps"][0]["attention"].as_array().unwrap().len() > 1);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(generation_body())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(body.matches("event: token").count(), 3);
        assert_eq!(body.matches("event: done").count(), 1);
        let done = body
            .split("event: done\n")
            .nth(1)
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .trim_start_matches("data: ");
        let streamed: Value = serde_json::from_str(done).unwrap();
        assert_eq!(streamed["generatedTokens"], value["generatedTokens"]);
    }
    #[tokio::test]
    async fn invalid_context_and_busy_runtime_are_rejected_before_generation() {
        let state = fixture();
        let app = router(state.clone(), None);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/generate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({"prompt":"abc","maxTokens":64}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let _permit = state.permit().unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/generate")
                    .header("content-type", "application/json")
                    .body(generation_body())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 429);
    }
    #[test]
    fn benchmark_runs_both_real_paths_and_enforces_fixed_output_count() {
        let state = fixture();
        let engine = state.engine.unwrap();
        let request = BenchmarkRequest {
            prompt: "abc".into(),
            max_tokens: 3,
            context_lengths: vec![2, 4],
            repeats: 2,
        };
        let result = run_benchmark(&engine, &request).unwrap();
        assert!(!result.tracing);
        assert_eq!(result.points.len(), 2);
        for point in result.points {
            assert!(point.tokens_match);
            assert_eq!(point.cache_on.generated_tokens, 3);
            assert_eq!(point.cache_off.generated_tokens, 3);
            assert_eq!(point.cache_on.decode_tokens, 2);
            assert!(point.cache_on.kv_cache_bytes > 0);
            assert_eq!(point.cache_off.kv_cache_bytes, 0);
            assert_eq!(point.cache_on.samples_ms_per_token.len(), 2);
            assert!(point.cache_on.decode_ms_per_token.is_finite());
        }
        assert!(validate_benchmark(
            &engine,
            &BenchmarkRequest {
                context_lengths: vec![2, 2],
                ..request.clone()
            }
        )
        .is_err());
        assert!(validate_benchmark(
            &engine,
            &BenchmarkRequest {
                max_tokens: 1,
                ..request
            }
        )
        .is_err());
    }
    #[test]
    fn benchmark_cancellation_stops_inside_a_run_without_partial_results() {
        let engine = fixture().engine.unwrap();
        let calls = std::cell::Cell::new(0usize);
        let result = run_benchmark_cancellable(
            &engine,
            &BenchmarkRequest {
                prompt: "abc".into(),
                max_tokens: 3,
                context_lengths: vec![2, 4],
                repeats: 2,
            },
            || {
                calls.set(calls.get() + 1);
                calls.get() >= 3
            },
        );
        assert!(result.unwrap_err().contains("cancelled"));
        assert_eq!(calls.get(), 3);
    }
    #[tokio::test]
    async fn streaming_benchmark_returns_actual_verified_measurements() {
        let response = router(fixture(), None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/benchmark/stream")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({"prompt":"abc","maxTokens":3,"contextLengths":[2,4],"repeats":1})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("event: done"));
        let data = text
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        let result: Value = serde_json::from_str(data).unwrap();
        assert_eq!(result["tracing"], false);
        assert_eq!(result["points"].as_array().unwrap().len(), 2);
        assert_eq!(result["points"][0]["tokensMatch"], true);
    }
    #[tokio::test]
    async fn malformed_json_and_oversized_http_bodies_have_structured_errors() {
        let app = router(fixture(), None);
        for (body, status) in [("{".to_owned(), 400), ("x".repeat(20 * 1024), 413)] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/generate")
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), status);
            let data: Value =
                serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                    .unwrap();
            assert_eq!(data["code"], "invalid_json");
            assert!(data["error"].as_str().unwrap().len() > 3);
        }
    }
    #[tokio::test]
    async fn api_root_never_falls_back_to_the_frontend() {
        let app = router(missing(), Some(PathBuf::from("../web/dist")));
        for path in ["/api", "/api/"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), 404, "{path}");
            assert_eq!(
                response.headers()["content-type"],
                "application/json",
                "{path}"
            );
        }
    }
}
