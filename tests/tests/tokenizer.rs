// Tokenizer tests require a real tokenizer.json to be present.
// These tests are skipped if no tokenizer is found.

use tiny_bit_core::tokenizer::Tokenizer;

fn load_tokenizer() -> Option<Tokenizer> {
    let path = std::path::Path::new("tokenizer.json");
    if !path.exists() {
        eprintln!("SKIP: tokenizer.json not found — run `tiny-bit download` first");
        return None;
    }
    Tokenizer::from_file(path).ok()
}

#[test]
fn test_encode_decode_roundtrip() {
    let Some(tok) = load_tokenizer() else { return };
    let text = "Hello world";
    let ids = tok.encode(text, false).unwrap();
    let decoded = tok.decode(&ids, true).unwrap();
    assert!(decoded.contains("Hello"), "decoded: {decoded}");
}

#[test]
fn test_special_tokens_present() {
    let Some(tok) = load_tokenizer() else { return };
    // All special token IDs should be non-zero and distinct
    let ids = [
        tok.tool_call_start_id,
        tok.tool_call_end_id,
        tok.tool_result_start_id,
        tok.tool_result_end_id,
        tok.assistant_token_id,
        tok.user_token_id,
        tok.system_token_id,
    ];
    let unique: std::collections::HashSet<u32> = ids.iter().cloned().collect();
    assert_eq!(unique.len(), ids.len(), "special token IDs are not all unique: {:?}", ids);
}

#[test]
fn test_chat_template() {
    let Some(tok) = load_tokenizer() else { return };
    let ids = tok.apply_chat_template(Some("You are helpful."), "Hello!").unwrap();
    assert!(!ids.is_empty(), "chat template produced empty token sequence");
    let text = tok.decode(&ids, false).unwrap();
    assert!(text.contains("<|user|>") || text.contains("<|assistant|>"),
        "chat template missing role tokens: {text}");
}
