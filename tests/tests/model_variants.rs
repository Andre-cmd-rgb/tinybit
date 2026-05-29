//! Guards for the V1.0 model-family layout and the shared prompt format.

use std::path::PathBuf;
use tinybit_core::config::ModelConfig;
use tinybit_core::tokenizer::{
    Profile, CODING_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT, ROLE_ASSISTANT_PREFIX,
    ROLE_SYSTEM_PREFIX, ROLE_USER_PREFIX, STOP_STRING_USER_TURN,
};

fn config_path(name: &str) -> PathBuf {
    // Resolve relative to this crate's manifest so it works regardless of CWD.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("configs").join(name)
}

const SIZES: &[&str] = &["nano", "micro", "small", "base"];

#[test]
fn all_eight_variants_parse_and_validate() {
    for size in SIZES {
        for suffix in ["", "-coding"] {
            let name = format!("{size}{suffix}.toml");
            let cfg = ModelConfig::from_file(&config_path(&name))
                .unwrap_or_else(|e| panic!("failed to load configs/{name}: {e}"));
            cfg.validate()
                .unwrap_or_else(|e| panic!("configs/{name} failed validation: {e}"));
            assert_eq!(cfg.vocab_size, 32008, "{name}: vocab_size must be 32008");
        }
    }
}

#[test]
fn coding_variants_match_their_general_sibling_architecture() {
    // The families differ only by training data + persona — the architecture
    // (hence checkpoint shapes) MUST be identical so checkpoints interchange.
    for size in SIZES {
        let general = ModelConfig::from_file(&config_path(&format!("{size}.toml"))).unwrap();
        let coding = ModelConfig::from_file(&config_path(&format!("{size}-coding.toml"))).unwrap();
        assert_eq!(general.param_count(), coding.param_count(), "{size}: param_count differs");
        assert_eq!(general.num_layers, coding.num_layers, "{size}: num_layers differs");
        assert_eq!(general.d_model, coding.d_model, "{size}: d_model differs");
        assert_eq!(general.d_ffn, coding.d_ffn, "{size}: d_ffn differs");
        assert_eq!(general.num_heads, coding.num_heads, "{size}: num_heads differs");
        assert_eq!(general.head_dim, coding.head_dim, "{size}: head_dim differs");
        assert_eq!(general.max_seq_len, coding.max_seq_len, "{size}: max_seq_len differs");
    }
}

#[test]
fn chat_template_constants_are_stable() {
    // Training-data formatting (scripts/prepare_data.sh) mirrors these exact
    // strings. If this test changes, the data script must change in lockstep.
    assert_eq!(ROLE_SYSTEM_PREFIX, "system:\n");
    assert_eq!(ROLE_USER_PREFIX, "\nuser:\n");
    assert_eq!(ROLE_ASSISTANT_PREFIX, "\nassistant:\n");
    assert_eq!(STOP_STRING_USER_TURN, "\nuser:");
    // The stop string must be a prefix of the user marker so generation halts at
    // the start of the next user turn.
    assert!(ROLE_USER_PREFIX.starts_with(STOP_STRING_USER_TURN));
}

#[test]
fn profile_parsing_and_prompts() {
    assert_eq!(Profile::parse("general").unwrap(), Profile::General);
    assert_eq!(Profile::parse("CODING").unwrap(), Profile::Coding);
    assert!(Profile::parse("nonsense").is_err());

    assert_eq!(Profile::General.default_system_prompt(), DEFAULT_SYSTEM_PROMPT);
    assert_eq!(Profile::Coding.default_system_prompt(), CODING_SYSTEM_PROMPT);

    // Filename-based inference used by chat/eval when --profile is absent.
    assert_eq!(Profile::from_config_path(&config_path("micro.toml")), Profile::General);
    assert_eq!(Profile::from_config_path(&config_path("micro-coding.toml")), Profile::Coding);
}
