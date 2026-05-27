// Tokenizer tests require a real tokenizer.json to be present.
// These tests are skipped if no tokenizer is found.

use tinybit_core::tokenizer::{Tokenizer, ROLE_ASSISTANT_PREFIX, ROLE_USER_PREFIX};

fn load_tokenizer() -> Option<Tokenizer> {
    let path = std::path::Path::new("tokenizer.json");
    if !path.exists() {
        eprintln!("SKIP: tokenizer.json not found — run `tinybit download` first");
        return None;
    }
    Tokenizer::from_file(path).ok()
}

fn load_tokenizer_with_vocab(vocab_size: usize) -> Option<Tokenizer> {
    let path = std::path::Path::new("tokenizer.json");
    if !path.exists() {
        eprintln!("SKIP: tokenizer.json not found — run `tinybit download` first");
        return None;
    }
    Tokenizer::from_file_with_vocab(path, vocab_size).ok()
}

#[test]
fn test_encode_decode_roundtrip() {
    let Some(tok) = load_tokenizer() else { return };
    let text = "Hello world";
    let ids = tok.encode(text, false).unwrap();
    let decoded = tok.decode(&ids, true).unwrap();
    assert!(decoded.contains("Hello"), "decoded: {decoded}");
}

/// With a generous vocab (default), tool-call special tokens *should* be
/// installable and distinct. (This is the "new training" scenario.)
#[test]
fn test_tool_tokens_when_vocab_has_room() {
    let Some(tok) = load_tokenizer() else { return };
    assert!(tok.supports_tool_tokens(), "tool tokens should fit when vocab_size is unconstrained");
    let ids = [
        tok.tool_call_start_id.unwrap(),
        tok.tool_call_end_id.unwrap(),
        tok.tool_result_start_id.unwrap(),
        tok.tool_result_end_id.unwrap(),
    ];
    let unique: std::collections::HashSet<u32> = ids.iter().cloned().collect();
    assert_eq!(unique.len(), ids.len(), "tool token ids are not unique: {ids:?}");
    for &id in &ids {
        assert!((id as usize) < tok.vocab_size(), "tool token id {id} exceeds vocab {}", tok.vocab_size());
    }
}

/// With a tight vocab cap (e.g. an older checkpoint trained on 32000 vocab),
/// `supports_tool_tokens()` must return false and no chat-template encoding
/// can ever produce an out-of-range id.
#[test]
fn test_tight_vocab_drops_tool_tokens_and_keeps_chat_safe() {
    let Some(tok) = load_tokenizer_with_vocab(32000) else { return };
    assert!(!tok.supports_tool_tokens());
    let ids = tok.apply_chat_template(Some("You are helpful."), "Hello!").unwrap();
    assert!(!ids.is_empty());
    for &id in &ids {
        assert!((id as usize) < 32000, "chat template produced id {id} >= 32000");
    }
}

#[test]
fn test_chat_template_uses_plain_text_role_markers() {
    let Some(tok) = load_tokenizer() else { return };
    let ids = tok.apply_chat_template(Some("You are helpful."), "Hello!").unwrap();
    assert!(!ids.is_empty(), "chat template produced empty token sequence");
    let text = tok.decode(&ids, false).unwrap();
    // Plain-text role markers — no <|user|>/<|assistant|> special tokens.
    assert!(text.contains(ROLE_USER_PREFIX.trim()),       "missing user marker: {text}");
    assert!(text.contains(ROLE_ASSISTANT_PREFIX.trim()),  "missing assistant marker: {text}");
}

/// Defensive: user input containing literal special-token strings must NOT
/// produce out-of-vocab ids when the tokenizer is capped.
#[test]
fn test_encode_drops_overflow_ids() {
    let Some(tok) = load_tokenizer_with_vocab(32000) else { return };
    // The literal `<|tool_call|>` would tokenize to >=32000 if installed as a
    // special token, but with a cap of 32000 it should be skipped and the
    // string tokenizes as ordinary text instead.
    let ids = tok.encode("<|tool_call|>payload<|end_tool_call|>", false).unwrap();
    for &id in &ids {
        assert!((id as usize) < 32000, "encode returned out-of-range id {id}");
    }
    assert!(!ids.is_empty());
}
