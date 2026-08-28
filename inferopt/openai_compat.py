"""OpenAI-compatible chat-completions rail (OpenRouter, Together, vLLM...).

Adapts the /v1/chat/completions shape to the internal representation so
call-site fingerprinting, cache analysis and reporting work unchanged.
"""
import json

from . import capture


def pseudo_body(body):
    """Map an OpenAI chat request to the anthropic-ish shape the
    fingerprinter expects (system text + messages + tool names)."""
    system_parts, msgs = [], []
    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):  # multimodal / content-block form
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text")
        else:
            text = content if isinstance(content, str) else ""
        if role in ("system", "developer"):
            system_parts.append(text)
        else:
            msgs.append({"role": role,
                         "content": [{"type": "text", "text": text}]})
    tools = []
    for t in body.get("tools") or []:
        fn = t.get("function") or {}
        if fn.get("name"):
            tools.append({"name": fn["name"]})
    out = {"system": "\n".join(system_parts), "messages": msgs}
    if tools:
        out["tools"] = tools
    return out


def _message_text(msg):
    """content, else serialized tool calls (structured-output parity with
    the other rails)."""
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content
                          if isinstance(b, dict) and b.get("type") == "text")
    if content:
        return content
    tools = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tools.append(capture.tool_sig(fn.get("name", "?"),
                                      fn.get("arguments", "")))
    return "\n".join(tools) or None


def extract_response(data):
    """-> (input, output, cache_read, cache_write, text)."""
    u = data.get("usage") or {}
    inp = u.get("prompt_tokens") or 0
    out = u.get("completion_tokens") or 0
    details = u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    # OpenAI-style automatic caching reports cached tokens INSIDE
    # prompt_tokens; split them so cost math doesn't double-count.
    inp = max(inp - cached, 0)
    written = (u.get("cache_creation_input_tokens")
               or details.get("cache_creation_tokens") or 0)
    choices = data.get("choices") or []
    text = _message_text((choices[0] or {}).get("message")) if choices else None
    return inp, out, cached, written, text


def apply_stream_event(line, state):
    """Feed one raw SSE line from a streaming chat-completions response."""
    line = line.strip()
    if not line.startswith(b"data:"):
        return
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return
    try:
        evt = json.loads(payload)
    except Exception:
        return
    u = evt.get("usage")
    if u:  # final chunk when stream_options.include_usage is set
        cached = (u.get("prompt_tokens_details") or {}).get(
            "cached_tokens") or 0
        state["input"] = max((u.get("prompt_tokens") or 0) - cached, 0)
        state["output"] = u.get("completion_tokens") or 0
        state["cr"] = cached
    for ch in evt.get("choices") or []:
        d = ch.get("delta") or {}
        if d.get("content"):
            state["text"].append(d["content"])
        for tc in d.get("tool_calls") or []:
            idx = tc.get("index", 0)
            while len(state["tools"]) <= idx:
                state["tools"].append({"name": "?", "parts": []})
            fn = tc.get("function") or {}
            if fn.get("name"):
                state["tools"][idx]["name"] = fn["name"]
            if fn.get("arguments"):
                state["tools"][idx]["parts"].append(fn["arguments"])
