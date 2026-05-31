#!/usr/bin/env python3
"""Validate a tinybit chat JSONL dataset (one {"messages":[...]} object per line).

Checks structure + the <|tool_call|>/<|tool_result|> protocol against the real
built-in tools. Usage: python scripts/validate_chat_jsonl.py <file.jsonl>
Exit 0 if all lines are usable (warnings allowed), 1 if any hard errors.
"""
import json, re, sys

VALID_ROLES = {"system", "user", "assistant"}
TOOLS = {
    "time":       set(),                                   # {}
    "calculator": {"expr"},
    "todos":      {"action", "text", "id"},
    "notes":      {"action", "title", "content", "query", "id"},
    "calendar":   {"action", "title", "date", "time", "notes", "from", "to", "id"},
}
SYS_CANON = ("You are tinybit, a small and efficient AI assistant built on the "
             "RWKV-7 architecture. You are helpful, concise, and honest.")

CALL = re.compile(r"<\|tool_call\|>(.*?)<\|end_tool_call\|>", re.S)
RESULT = re.compile(r"<\|tool_result\|>(.*?)<\|end_tool_result\|>", re.S)

def check_assistant_tools(content, errs, warns, i):
    # marker balance
    for open_m, close_m in (("<|tool_call|>", "<|end_tool_call|>"),
                            ("<|tool_result|>", "<|end_tool_result|>")):
        if content.count(open_m) != content.count(close_m):
            errs.append(f"line {i}: unbalanced {open_m}/{close_m}")
    calls = CALL.findall(content)
    results = RESULT.findall(content)
    if calls and not results:
        warns.append(f"line {i}: tool_call without a tool_result (model would stall at inference)")
    for raw in calls:
        try:
            obj = json.loads(raw)
        except Exception as e:
            errs.append(f"line {i}: tool_call JSON invalid: {e!r} :: {raw[:80]}")
            continue
        if not isinstance(obj, dict) or "tool" not in obj or "args" not in obj:
            errs.append(f"line {i}: tool_call must be {{'tool','args'}} :: {raw[:80]}")
            continue
        tool, args = obj["tool"], obj["args"]
        if tool not in TOOLS:
            errs.append(f"line {i}: unknown tool {tool!r} (allowed: {sorted(TOOLS)})")
            continue
        if not isinstance(args, dict):
            errs.append(f"line {i}: args must be an object for tool {tool!r}")
            continue
        extra = set(args) - TOOLS[tool]
        if extra:
            warns.append(f"line {i}: tool {tool!r} has unexpected arg(s) {sorted(extra)}")

def main(path):
    errs, warns = [], []
    n = n_tool = n_sys = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                errs.append(f"line {i}: not valid JSON: {e!r}")
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or not msgs:
                errs.append(f"line {i}: missing/empty 'messages' array")
                continue
            roles = [m.get("role") for m in msgs]
            has_tool = False
            for m in msgs:
                r, c = m.get("role"), m.get("content")
                if r not in VALID_ROLES:
                    errs.append(f"line {i}: bad role {r!r}")
                if not isinstance(c, str) or not c.strip():
                    errs.append(f"line {i}: role {r!r} has empty/non-string content")
                    continue
                if r == "system" and c.strip() != SYS_CANON:
                    warns.append(f"line {i}: non-canonical system prompt")
                if r == "assistant":
                    if "<|tool_call|>" in c:
                        has_tool = True
                    check_assistant_tools(c, errs, warns, i)
            if "system" in roles:
                n_sys += 1
            if has_tool:
                n_tool += 1
            # structural sanity: must contain at least one user and one assistant,
            # and (ignoring a leading system) should end on an assistant turn.
            if "user" not in roles or "assistant" not in roles:
                errs.append(f"line {i}: needs at least one user and one assistant turn")
            if roles and roles[-1] != "assistant":
                warns.append(f"line {i}: conversation does not end on assistant turn")

    print(f"entries: {n}")
    print(f"  with a tool call : {n_tool} ({100*n_tool//max(n,1)}%)")
    print(f"  with system turn : {n_sys} ({100*n_sys//max(n,1)}%)")
    print(f"  hard errors      : {len(errs)}")
    print(f"  warnings         : {len(warns)}")
    for e in errs[:25]:
        print("  ERROR  " + e)
    for w in warns[:15]:
        print("  warn   " + w)
    if len(errs) > 25 or len(warns) > 15:
        print(f"  ... ({len(errs)} errors, {len(warns)} warnings total)")
    return 1 if errs else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
