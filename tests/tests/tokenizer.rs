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

/// The shipped LLaMA tokenizer does NOT define the `<|tool_*|>` markers, and we
/// deliberately do not mint them: data prep trains the model on the markers'
/// ordinary BPE spelling, so minting single token ids here would inject
/// untrained embeddings when a tool result is encoded back into context. The
/// contract is therefore: `supports_tool_tokens()` is false, but the markers
/// still round-trip through encode/decode as text (that is how `parse_tool_call`
/// finds them) and never produce an out-of-range id.
#[test]
fn test_tool_markers_round_trip_as_plain_text() {
    let Some(tok) = load_tokenizer() else { return };
    assert!(
        !tok.supports_tool_tokens(),
        "shipped tokenizer must not mint tool tokens — the model trains on the BPE spelling"
    );
    let marker = "<|tool_call|>{\"tool\":\"x\"}<|end_tool_call|>";
    let ids = tok.encode(marker, false).unwrap();
    assert!(!ids.is_empty());
    for &id in &ids {
        assert!((id as usize) < tok.vocab_size(), "marker token id {id} exceeds vocab {}", tok.vocab_size());
    }
    let decoded = tok.decode(&ids, false).unwrap();
    assert!(decoded.contains("<|tool_call|>"),     "marker did not round-trip: {decoded}");
    assert!(decoded.contains("<|end_tool_call|>"), "marker did not round-trip: {decoded}");
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

/// Minimal LLaMA-style tokenizer (WordLevel vocab + the same
/// Replace→ByteFallback→Fuse→Strip decoder stack the real file uses), written
/// to a temp file so `IncrementalDecoder` gets real coverage without the
/// downloaded tokenizer.json. Ids: 1 "▁hello", 2 "▁world", 3/4 the two bytes
/// of "é" (byte-fallback), 5 "!", 6 "▁<", 7 "|user", 8 ":".
fn mini_tokenizer() -> Tokenizer {
    let json = r#"{
      "version": "1.0",
      "truncation": null, "padding": null,
      "added_tokens": [
        {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null, "pre_tokenizer": null, "post_processor": null,
      "decoder": {
        "type": "Sequence",
        "decoders": [
          {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
          {"type": "ByteFallback"},
          {"type": "Fuse"},
          {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]
      },
      "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "▁hello": 1, "▁world": 2,
                  "<0xC3>": 3, "<0xA9>": 4, "!": 5, "▁<": 6,
                  "|user": 7, ":": 8},
        "unk_token": "<unk>"
      }
    }"#;
    let path = std::env::temp_dir().join(format!("tinybit-mini-tok-{}.json", std::process::id()));
    std::fs::write(&path, json).expect("write mini tokenizer");
    let tok = Tokenizer::from_file(&path).expect("load mini tokenizer");
    let _ = std::fs::remove_file(&path);
    tok
}

/// The incremental decoder's chunk concatenation must equal a one-shot decode
/// of the full id sequence — including across the leading-space carry between
/// `▁`-prefixed pieces and through byte-fallback tokens that spell one code
/// point over multiple ids.
#[test]
fn test_incremental_decoder_matches_full_decode_mini() {
    let tok = mini_tokenizer();
    let cases: Vec<Vec<u32>> = vec![
        vec![1, 2, 5],          // "hello world!"
        vec![3, 4],             // byte-fallback "é"
        vec![1, 3, 4, 2],       // mixed: "helloé world"
        vec![6, 7, 8, 1],       // marker-ish "<|user:" shapes
        vec![2],                // single token
        vec![5, 5, 5],          // no-space pieces
        vec![1, 2, 1, 2, 3, 4, 5, 6, 7, 8],
    ];
    for ids in cases {
        for skip_special in [false, true] {
            let full = tok.decode(&ids, skip_special).expect("full decode");
            let mut stream = tok.decode_stream(skip_special);
            let mut acc = String::new();
            for &id in &ids {
                if let Some(chunk) = stream.step(id).expect("step") {
                    acc.push_str(&chunk);
                }
            }
            assert_eq!(acc, full, "ids {ids:?} skip_special={skip_special}");
        }
    }
}

/// An incomplete byte-fallback sequence is held back (None), then emitted in
/// one piece when the code point completes — never a replacement char.
#[test]
fn test_incremental_decoder_holds_back_partial_utf8() {
    let tok = mini_tokenizer();
    let mut stream = tok.decode_stream(false);
    assert_eq!(stream.step(3).expect("step"), None); // first byte of é
    assert_eq!(stream.step(4).expect("step").as_deref(), Some("é"));
}

/// Same property against the real LLaMA tokenizer (skipped without
/// tokenizer.json): random id sequences plus the strings the generation loop
/// scans for (tool markers, stop string, multi-byte text).
#[test]
fn test_incremental_decoder_matches_full_decode_real() {
    let Some(tok) = load_tokenizer() else { return };
    let mut cases: Vec<Vec<u32>> = Vec::new();
    for text in [
        "Hello world",
        "<|tool_call|>{\"tool\":\"calculator\",\"args\":{\"expr\":\"1+2\"}}<|end_tool_call|>",
        "done.\nuser:\nnext question",
        "naïve café — résumé ✓ 日本語",
    ] {
        cases.push(tok.encode(text, false).expect("encode"));
    }
    // Deterministic pseudo-random id soup (hits byte-fallback + odd pieces).
    let mut x = 0x2545F491u64;
    let vocab = tok.vocab_size() as u64;
    for len in [1usize, 7, 64, 300] {
        let mut ids = Vec::with_capacity(len);
        for _ in 0..len {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ids.push(((x >> 33) % vocab) as u32);
        }
        cases.push(ids);
    }
    for ids in cases {
        for skip_special in [false, true] {
            let full = tok.decode(&ids, skip_special).expect("full decode");
            let mut stream = tok.decode_stream(skip_special);
            let mut acc = String::new();
            for &id in &ids {
                if let Some(chunk) = stream.step(id).expect("step") {
                    acc.push_str(&chunk);
                }
            }
            assert_eq!(acc, full, "{} ids, skip_special={skip_special}", ids.len());
        }
    }
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
