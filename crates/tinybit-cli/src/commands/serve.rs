use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tinybit_core::config::ModelConfig;
use tinybit_infer::{engine::InferenceEngine, session::Session};
use tokio::net::TcpListener;

#[derive(Args)]
pub struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    #[arg(long, default_value = "configs/small.toml")]
    pub config: std::path::PathBuf,

    #[arg(long, default_value = "models/tinybit-small.safetensors")]
    pub model: std::path::PathBuf,

    #[arg(long, default_value = "tokenizer.json")]
    pub tokenizer: std::path::PathBuf,

    #[arg(long, default_value = "data/")]
    pub data_dir: std::path::PathBuf,
}

struct AppState {
    engine: Arc<InferenceEngine>,
    config: ModelConfig,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(default = "default_temp")]
    temperature: f64,
    #[serde(default = "default_max")]
    max_tokens: usize,
}

fn default_temp() -> f64 { 0.7 }
fn default_max() -> usize { 512 }

#[derive(Deserialize, Serialize, Clone)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: OpenAiMessage,
    finish_reason: &'static str,
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let device = InferenceEngine::auto_device();
    let mut session = match Session::new(&state.config, &device) {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    // Extract the last user message
    let user_msg = req.messages.iter().rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let response = match state.engine.chat_turn(&user_msg, &mut session, None) {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", chrono::Utc::now().timestamp()),
        object: "chat.completion",
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: OpenAiMessage { role: "assistant".to_string(), content: response },
            finish_reason: "stop",
        }],
    }).into_response()
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo { id: "tinybit-small".to_string(), object: "model" }],
    })
}

async fn health() -> &'static str { "ok" }

#[derive(Serialize)]
struct Metrics {
    uptime_secs: u64,
    model: &'static str,
}

static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

async fn metrics() -> Json<Metrics> {
    let start = START.get_or_init(std::time::Instant::now);
    Json(Metrics {
        uptime_secs: start.elapsed().as_secs(),
        model: "tinybit-small",
    })
}

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    START.get_or_init(std::time::Instant::now);
    let config = ModelConfig::from_file(&args.config)?;
    let device = InferenceEngine::auto_device();
    let engine = Arc::new(InferenceEngine::new(
        &args.model,
        config.clone(),
        &args.tokenizer,
        &args.data_dir,
        device,
    )?);

    let state = Arc::new(AppState { engine, config });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    println!("tinybit server listening on http://{addr}");
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
