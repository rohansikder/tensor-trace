use clap::{Parser, Subcommand, ValueEnum};
use std::{net::IpAddr, path::PathBuf};
use tensor_trace_engine::{Engine, GenerationRequest};
use tensor_trace_server::{router, run_benchmark, AppState, BenchmarkRequest};

#[derive(Parser)]
#[command(
    name = "tensor-trace",
    version,
    about = "Inspect a CPU language model one token at a time"
)]
struct Cli {
    #[arg(long, global = true, default_value = "models/stories15M.bin")]
    model: PathBuf,
    #[arg(long, global = true, default_value = "models/tokenizer.bin")]
    tokenizer: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Serve the local JSON/SSE API and compiled console.
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        #[arg(long, default_value_t = 8787)]
        port: u16,
        #[arg(long, default_value = "web/dist")]
        web_dir: PathBuf,
        #[arg(long,default_value_t=1,value_parser=clap::value_parser!(u8).range(1..=2))]
        concurrent_jobs: u8,
    },
    /// Generate text using the Rust runtime, without the browser.
    Generate {
        #[arg(long, default_value = "Once upon a time")]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, value_enum, default_value = "on")]
        cache: CacheMode,
        #[arg(long)]
        json: bool,
    },
    /// Dump final logits and a selected attention head for explicit token IDs.
    Inspect {
        #[arg(long, value_delimiter = ',', required = true)]
        tokens: Vec<usize>,
        #[arg(long, value_enum, default_value = "on")]
        cache: CacheMode,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 0)]
        head: usize,
        /// Inspection is always JSON; retained for explicit shell scripts.
        #[arg(long)]
        json: bool,
    },
    /// Compare cached/uncached decoding on identical tokens, with tracing off.
    #[command(alias = "benchmark")]
    Bench {
        #[arg(long, default_value = "Once upon a time")]
        prompt: String,
        #[arg(long, default_value_t = 8)]
        max_tokens: usize,
        #[arg(long, value_delimiter = ',', default_value = "8,16,32")]
        context_lengths: Vec<usize>,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
    },
    /// Print actual checkpoint dimensions and runtime metadata.
    Model,
}
#[derive(Clone, Copy, ValueEnum)]
enum CacheMode {
    On,
    Off,
}
impl CacheMode {
    fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
async fn run() -> Result<(), String> {
    let cli = Cli::parse();
    if let Command::Serve {
        host,
        port,
        web_dir,
        concurrent_jobs,
    } = cli.command
    {
        let state = AppState::load(&cli.model, &cli.tokenizer, concurrent_jobs as usize);
        if let Some(error) = &state.load_error {
            eprintln!("Model unavailable: {error}");
        }
        let listener = tokio::net::TcpListener::bind((host, port))
            .await
            .map_err(|error| error.to_string())?;
        eprintln!("TensorTrace is listening at http://{host}:{port}");
        return axum::serve(listener, router(state, Some(web_dir)))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .map_err(|error| error.to_string());
    }
    let engine = Engine::load(&cli.model, &cli.tokenizer)?;
    match cli.command {
        Command::Generate {
            prompt,
            max_tokens,
            temperature,
            top_p,
            seed,
            cache,
            json,
        } => {
            let request = GenerationRequest {
                prompt,
                max_tokens,
                temperature,
                top_p,
                seed,
                cache: cache.enabled(),
                trace: false,
                ..GenerationRequest::default()
            };
            let result = engine.generate(&request)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
                );
            } else {
                println!("{}", result.text);
                eprintln!("prefill {:.2} ms · decode {:.2} ms/token · {:.2} tokens/sec · active reusable KV {} bytes",result.stats.prefill_ms,result.stats.decode_ms_per_token,result.stats.tokens_per_second,result.stats.kv_cache_bytes);
            }
        }
        Command::Inspect {
            tokens,
            cache,
            layer,
            head,
            ..
        } => {
            let result = engine.inspect_tokens(&tokens, cache.enabled(), layer, head)?;
            println!(
                "{}",
                serde_json::to_string(&result).map_err(|error| error.to_string())?
            );
        }
        Command::Bench {
            prompt,
            max_tokens,
            context_lengths,
            repeats,
        } => {
            let request = BenchmarkRequest {
                prompt,
                max_tokens,
                context_lengths,
                repeats,
            };
            let result = run_benchmark(&engine, &request)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
            );
        }
        Command::Model => println!(
            "{}",
            serde_json::to_string_pretty(&engine.metadata()).map_err(|error| error.to_string())?
        ),
        Command::Serve { .. } => unreachable!(),
    }
    Ok(())
}
