use tinybit_core::{config::ModelConfig, state::InferenceState};
use chrono::{DateTime, Utc};
use candle_core::Device;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role:    Role,
    pub content: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Role {
    User, Assistant, ToolCall, ToolResult, System,
}

pub struct Session {
    pub id:            String,
    pub state:         InferenceState,
    pub history:       Vec<ChatMessage>,
    pub system_prompt: String,
    pub created_at:    DateTime<Utc>,
}

impl Session {
    pub fn new(config: &ModelConfig, device: &Device) -> anyhow::Result<Self> {
        Ok(Self {
            id:            uuid(),
            state:         InferenceState::zeros(config, device)?,
            history:       Vec::new(),
            system_prompt: String::new(),
            created_at:    Utc::now(),
        })
    }

    pub fn reset_state(&mut self, config: &ModelConfig) -> anyhow::Result<()> {
        self.state = InferenceState::zeros(config, &self.state.device)?;
        Ok(())
    }

    pub fn save(&self, dir: &std::path::Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        let history_json = serde_json::to_string_pretty(&self.history)?;
        std::fs::write(dir.join("history.json"), history_json)?;
        self.state.save(&dir.join("state.safetensors"))?;
        std::fs::write(dir.join("system_prompt.txt"), &self.system_prompt)?;
        Ok(())
    }

    pub fn load(dir: &std::path::Path, device: &Device) -> anyhow::Result<Self> {
        let history: Vec<ChatMessage> = serde_json::from_str(
            &std::fs::read_to_string(dir.join("history.json"))?
        )?;
        let state = InferenceState::load(&dir.join("state.safetensors"), device)?;
        let system_prompt = std::fs::read_to_string(dir.join("system_prompt.txt"))
            .unwrap_or_default();
        Ok(Self {
            id: uuid(),
            state,
            history,
            system_prompt,
            created_at: Utc::now(),
        })
    }
}

fn uuid() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    format!("{n:x}")
}
