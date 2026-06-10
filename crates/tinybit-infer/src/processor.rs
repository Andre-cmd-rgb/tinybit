use crate::engine::{GenStats, InferenceEngine};
use crate::sampler::sample;
use candle_core::{DType, Tensor};
use std::time::Instant;
use tinybit_core::state::InferenceState;
use tinybit_core::tokenizer::STOP_STRING_USER_TURN;
use tinybit_tools::parser::{
    format_tool_result, parse_tool_call, strip_tool_markers, ToolCall, CALL_END,
};
use tinybit_tools::ToolOutput;

const MAX_TOOL_ROUNDS: usize = 8;

/// How the chat loop decides whether the model is allowed to emit a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolMode {
    /// Allow a tool call only when the user's message plausibly needs one
    /// (`message_needs_tools`). This is the default: a tiny model otherwise
    /// fires tools reflexively on greetings and chit-chat.
    #[default]
    Auto,
    /// Never gate — the raw model decides (it over-fires; useful for eval/debug).
    Always,
    /// Never allow a tool call — pure conversation.
    Never,
}

impl ToolMode {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(ToolMode::Auto),
            "always" | "on" => Ok(ToolMode::Always),
            "never" | "off" => Ok(ToolMode::Never),
            other => anyhow::bail!("unknown tool mode '{other}' (expected: auto | always | never)"),
        }
    }
}

/// Cheap, deliberately conservative heuristic: does this user message plausibly
/// call for one of the built-in tools? Biased toward `false` — over-firing is
/// the failure mode we're guarding against, so when in doubt we suppress and let
/// the model answer in words. Matches on the lowercased message.
pub fn message_needs_tools(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();

    // Greetings / acknowledgements are never tool turns, even though some start
    // with "what's…" (which the factual-lookup rule below would otherwise arm).
    let bare = m.trim().trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
    if [
        "hi", "hey", "hello", "yo", "sup", "whats up", "what's up", "what up",
        "whatsup", "hiya", "good morning", "good afternoon", "good evening",
        "goodnight", "good night", "thanks", "thank you", "ty", "ok", "okay",
        "cool", "nice", "lol", "bye", "goodbye",
    ]
    .contains(&bare)
    {
        return false;
    }

    let has_digit = m.bytes().any(|b| b.is_ascii_digit());

    // calculator: an explicit `digit op digit` (ignoring spaces) or an `=`, or a
    // number paired with an arithmetic word.
    let compact: Vec<u8> = m.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let arithmetic = m.contains('=')
        || compact.windows(3).any(|w| {
            w[0].is_ascii_digit()
                && matches!(w[1], b'+' | b'-' | b'*' | b'/' | b'^' | b'x')
                && w[2].is_ascii_digit()
        });
    let math_word = ["calculat", "comput", "plus", "minus", "times", "multipl",
                     "divid", "sqrt", "square root", "percent"]
        .iter().any(|w| m.contains(w));
    if arithmetic || (has_digit && math_word) {
        return true;
    }

    // time / date — cover the common natural phrasings without matching
    // incidental uses of "time" ("a long time", "do you have time").
    if ["what time", "what day", "time is it", "current time", "the time today",
        "time today", "time right now", "time now", "what is the time",
        "what's the time", "whats the time", "tell me the time",
        "today's date", "date today", "what is the date", "what's the date",
        "whats the date", "what date"]
        .iter().any(|w| m.contains(w)) {
        return true;
    }

    // todos — require an action verb, not just the word "list" (so "write a
    // to-do list …" is answered in prose, not by calling the tool).
    let todo_word = m.contains("todo") || m.contains("to-do") || m.contains(" task");
    let todo_action = ["add ", "remove", "delete", "complete", "mark ", "finish",
                       "cross off", "show my", "list my", "what's on", "whats on"]
        .iter().any(|w| m.contains(w));
    if (todo_word && todo_action) || m.contains("remind me") {
        return true;
    }

    // notes
    if (m.contains("note")
        && ["save", "take", "write", "add", "show", "find", "search", "read", "recall"]
            .iter().any(|w| m.contains(w)))
        || m.contains("write down") || m.contains("make a note")
    {
        return true;
    }

    // calendar
    if ["calendar", "schedule", "appointment", "my events", "add event",
        "meeting on", "remind me on"]
        .iter().any(|w| m.contains(w)) {
        return true;
    }

    // lookup over the user's local documents ("search my docs for the deploy
    // schedule", "what do my files say about the cabin wifi").
    if (m.contains("my docs") || m.contains("my documents") || m.contains("my files")
        || m.contains("my notes"))
        && ["search", "find", "look", "check", "what", "say", "anything"]
            .iter().any(|w| m.contains(w))
    {
        return true;
    }

    // lookup — general factual questions ("what is the capital of france",
    // "who invented the telephone", "how many planets"), but NOT questions about
    // tinybit itself (those are identity, answered directly, not looked up).
    let about_self = m.contains("you") || m.contains("your") || m.contains("tinybit");
    let factual = [
        "what is", "what's", "whats", "what are", "what was", "what does",
        "who is", "who was", "who invented", "who wrote", "who discovered",
        "who painted", "who created", "who built", "when did", "when was",
        "where is", "where are", "how many", "how tall", "how big", "how far",
        "how old", "how deep", "how high", "how fast", "capital of",
        "tell me about", "define ",
    ]
    .iter().any(|w| m.contains(w));
    if factual && !about_self {
        return true;
    }

    false
}

/// Given the visible text decoded so far and how many chars were already
/// streamed, return the next chunk safe to emit and the new emitted count.
/// `strip_tool_markers` removes the tool protocol — a started tool call (and
/// everything after it, rendered separately by `render_tool_use`) plus any
/// malformed marker the model echoes — so `emitted` is counted against that
/// cleaned, monotonic stream. The last `holdback` chars are withheld so a
/// partial stop string (`\nuser:`) isn't streamed before we detect it; pass
/// `holdback = 0` to flush the remainder once the text is final.
fn next_chunk(visible: &str, emitted: usize, holdback: usize) -> (String, usize) {
    let cleaned = strip_tool_markers(visible);
    let chars: Vec<char> = cleaned.chars().collect();
    let safe = chars.len().saturating_sub(holdback);
    if safe > emitted {
        (chars[emitted..safe].iter().collect(), safe)
    } else {
        (String::new(), emitted)
    }
}

/// Human-facing rendering of an executed tool call (also what lands in the saved
/// transcript). The raw `<|tool_call|>…<|tool_result|>…` protocol is internal and
/// never shown; this is the clean affordance, e.g. `[calculator {"expr":"1+2"} -> 3]`.
fn render_tool_use(call: &ToolCall, output: &ToolOutput) -> String {
    let status = if output.is_error { "error: " } else { "" };
    format!(" [{} {} -> {}{}] ", call.tool, call.args, status, output.content.trim())
}

pub struct ToolProcessor<'a> {
    engine: &'a InferenceEngine,
    max_rounds: usize,
}

impl<'a> ToolProcessor<'a> {
    pub fn new(engine: &'a InferenceEngine) -> Self {
        Self { engine, max_rounds: MAX_TOOL_ROUNDS }
    }

    /// `tool_ban` is the set of token ids the sampler must never emit this turn.
    /// The tool gate passes the token that begins `<|tool_call|>` here (and so
    /// blocks the whole marker) when the turn shouldn't use a tool; pass `&[]` to
    /// let the model emit tool calls freely.
    pub fn run(
        &self,
        encoded_prompt: &[u32],
        state: &mut InferenceState,
        tool_ban: &[u32],
        mut on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<(String, GenStats)> {
        let eng = self.engine;
        let mut full_output = String::new();
        let mut current_ids: Vec<u32> = encoded_prompt.to_vec();
        let mut stats = GenStats { prompt_tokens: encoded_prompt.len(), ..Default::default() };

        // Stream tokens as they're produced when a sink is provided. We hold
        // back the length of the stop string so a partial `\nuser:` is never
        // emitted before the loop detects it and stops.
        let streaming = on_token.is_some();
        // Withhold a partial stop string from the tail until we can detect it.
        // (Tool markers don't need holdback — `strip_tool_markers` removes them
        // at any length, so streaming stays low-latency.)
        let holdback = STOP_STRING_USER_TURN.chars().count();

        // Prefill every prompt token EXCEPT the last; the last token's forward
        // happens in the decode loop and produces the first prediction. Feeding
        // it in both phases would apply it to the recurrent state twice, so the
        // first generated token would be conditioned on a duplicated final
        // prompt token.
        let (prefill_ids, last_id) = match encoded_prompt.split_last() {
            Some((last, head)) => (head, *last),
            None => (&[][..], eng.tokenizer.bos_token_id),
        };
        let t_prefill = Instant::now();
        if !prefill_ids.is_empty() {
            // One chunked sequence forward instead of a per-token loop — every
            // projection runs as a single GEMM over the prompt rows (decision 20)
            // and the state ends exactly as token-by-token stepping would
            // (parity pinned by test_prefill_matches_step).
            let ids = Tensor::from_vec(prefill_ids.to_vec(), (1, prefill_ids.len()), &eng.device)?
                .to_dtype(DType::U32)?;
            eng.model.forward_prefill(&ids, state)?;
        }
        stats.prefill_secs = t_prefill.elapsed().as_secs_f64();

        let mut prev_id = last_id;

        let t_decode = Instant::now();
        for _round in 0..self.max_rounds {
            let mut round_tokens: Vec<u32> = Vec::new();
            let mut hit_stop = false;
            let mut emitted = 0usize; // chars of this round already streamed

            // Incremental decoders for this round: `partial` mirrors
            // decode(&round_tokens, false) for the per-token marker/stop scan,
            // `visible` mirrors decode(&round_tokens, true) for streaming.
            // Each token costs O(1) amortized instead of re-decoding the whole
            // round buffer (which made a turn O(n²) in generated length).
            let mut scan_stream = eng.tokenizer.decode_stream(false);
            let mut partial = String::new();
            let mut vis_stream = eng.tokenizer.decode_stream(true);
            let mut visible = String::new();

            for _ in 0..eng.params.max_new_tokens {
                let tid =
                    Tensor::from_vec(vec![prev_id], (1, 1), &eng.device)?.to_dtype(DType::U32)?;
                let logits = eng.model.forward_step(&tid, state)?;
                let next_id = sample(&logits, &eng.params, &current_ids, tool_ban)?;
                stats.gen_tokens += 1;

                if next_id == eng.tokenizer.eos_token_id {
                    let text =
                        strip_tool_markers(&eng.tokenizer.decode(&round_tokens, true)?).into_owned();
                    let trimmed = text.trim_end();
                    if streaming {
                        let (chunk, _) = next_chunk(trimmed, emitted, 0);
                        if let Some(cb) = on_token.as_deref_mut() {
                            cb(&chunk);
                        }
                    }
                    full_output.push_str(trimmed);
                    stats.decode_secs = t_decode.elapsed().as_secs_f64();
                    return Ok((full_output, stats));
                }

                round_tokens.push(next_id);
                current_ids.push(next_id);
                prev_id = next_id;

                if streaming {
                    if let Ok(Some(chunk)) = vis_stream.step(next_id) {
                        visible.push_str(&chunk);
                    }
                    let (chunk, new_emitted) = next_chunk(&visible, emitted, holdback);
                    if !chunk.is_empty() {
                        if let Some(cb) = on_token.as_deref_mut() {
                            cb(&chunk);
                        }
                        emitted = new_emitted;
                    }
                }

                // Check completion conditions EVERY token. tinybit's `<|tool_*|>`
                // markers are ordinary multi-token BPE text (not single vocab
                // ids — see tokenizer::resolve_marker), so only a per-token scan
                // catches `<|end_tool_call|>` the instant it completes. The old
                // every-4-tokens scan detected it up to 3 tokens late, and those
                // 1–3 extra tokens — the model's own start of a `<|tool_result|>`
                // — were forward_step'd into the recurrent state BEFORE we
                // injected the real result, corrupting the state and tipping the
                // model into post-call garbage. `partial` is maintained
                // incrementally (see scan_stream above) and the markers end in
                // ASCII, so the completing token always surfaces its chunk in
                // the same step — detection latency is unchanged.
                {
                    if let Ok(Some(chunk)) = scan_stream.step(next_id) {
                        partial.push_str(&chunk);
                    }

                    // Complete tool call?
                    if partial.contains(CALL_END) {
                        if let Some((call, before, _after)) = parse_tool_call(&partial) {
                            // Flush any text that preceded the call.
                            if streaming {
                                let (chunk, _) = next_chunk(before, emitted, 0);
                                if let Some(cb) = on_token.as_deref_mut() {
                                    cb(&chunk);
                                }
                            }
                            full_output.push_str(&strip_tool_markers(before));
                            let result = eng.tools.execute(&call).unwrap_or_else(|e| {
                                ToolOutput::err(e.to_string())
                            });
                            // Show only a clean rendering; the raw protocol below
                            // is fed to the model but never displayed or saved.
                            let rendered = render_tool_use(&call, &result);
                            full_output.push_str(&rendered);
                            if streaming {
                                if let Some(cb) = on_token.as_deref_mut() {
                                    cb(&rendered);
                                }
                            }
                            // Inject the marker-wrapped result back into context —
                            // the format the model was trained on. (encode is
                            // vocab-safe and drops any overflow ids.) Feed all but
                            // the LAST result token: the next decode iteration
                            // forward-steps `prev_id`, so feeding the last one here
                            // too would apply it to the recurrent state twice
                            // (same invariant as the prompt prefill above).
                            let result_str = format_tool_result(&result);
                            let result_ids = eng.tokenizer.encode(&result_str, false)?;
                            if let Some((&last, head)) = result_ids.split_last() {
                                if !head.is_empty() {
                                    let ids =
                                        Tensor::from_vec(head.to_vec(), (1, head.len()), &eng.device)?
                                            .to_dtype(DType::U32)?;
                                    eng.model.forward_prefill(&ids, state)?;
                                }
                                current_ids.extend_from_slice(&result_ids);
                                prev_id = last;
                            }
                            round_tokens.clear();
                            break; // next round
                        }
                    }

                    // Next user turn marker → stop.
                    if partial.contains(STOP_STRING_USER_TURN) {
                        hit_stop = true;
                        break;
                    }
                }
            }

            if !round_tokens.is_empty() {
                let mut text =
                    strip_tool_markers(&eng.tokenizer.decode(&round_tokens, true)?).into_owned();
                if let Some(idx) = text.find(STOP_STRING_USER_TURN) {
                    text.truncate(idx);
                }
                let trimmed = text.trim_end();
                if streaming {
                    let (chunk, _) = next_chunk(trimmed, emitted, 0);
                    if let Some(cb) = on_token.as_deref_mut() {
                        cb(&chunk);
                    }
                }
                full_output.push_str(trimmed);
                if hit_stop {
                    break;
                }
                // Otherwise — we hit max_new_tokens without a tool call. Stop.
                break;
            }
        }
        stats.decode_secs = t_decode.elapsed().as_secs_f64();
        Ok((full_output, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_suppresses_chitchat_and_identity() {
        for m in [
            "hi", "hello there", "wtf bro", "df", "oh come on", "thanks",
            "who are you?", "what can you do?", "tell me a fun fact about dogs",
            "i have 3 cats, any tips?",
            "write a short to-do list for moving house", // 'list' alone must NOT arm
            "i exercise and read", "name a few hobbies",
            "do you have time to help?", "i had a great time", // incidental "time"
            "what's up", "whats up", "what can you do?", // identity / greeting, not lookup
            "what is your name?", "what are you?",        // about tinybit → not lookup
        ] {
            assert!(!message_needs_tools(m), "should NOT arm tools: {m:?}");
        }
    }

    #[test]
    fn gate_allows_genuine_tool_requests() {
        for m in [
            "1+324", "213-134", "15*8", "what is 15 times 8?", "calculate 99/3",
            "what time is it?", "what day is it today?",
            "what is the time today", "what's the date?", "tell me the time",
            "add milk to my todos", "remind me to call the dentist",
            "save a note about the meeting", "what's on my calendar?",
            // factual lookup
            "what is the capital of france?", "who invented the telephone",
            "how many planets are there", "tell me about the great wall of china",
            "what is the largest ocean?", "what's the third planet?",
            // lookup over local documents
            "search my docs for the deploy schedule",
            "what do my files say about the cabin wifi?",
            "check my documents for the meeting agenda",
        ] {
            assert!(message_needs_tools(m), "SHOULD arm tools: {m:?}");
        }
    }

    #[test]
    fn tool_mode_parses() {
        assert_eq!(ToolMode::parse("auto").unwrap(), ToolMode::Auto);
        assert_eq!(ToolMode::parse("ALWAYS").unwrap(), ToolMode::Always);
        assert_eq!(ToolMode::parse("never").unwrap(), ToolMode::Never);
        assert!(ToolMode::parse("bogus").is_err());
    }
}
