"""Response-content extraction shared by both rails.

Tool-calling responses (e.g. LangChain .with_structured_output) contain no
text blocks - the payload lives in tool_use.input. For logging and replay
comparison we serialize those as '[tool_use <name>] <sorted-json>' so
structured outputs are first-class comparable responses, not NULLs.
"""
import json


def tool_sig(name, input_obj):
    if isinstance(input_obj, str):
        s = input_obj
        try:
            s = json.dumps(json.loads(input_obj), sort_keys=True,
                           ensure_ascii=False)
        except Exception:
            pass
    else:
        s = json.dumps(input_obj, sort_keys=True, ensure_ascii=False)
    return f"[tool_use {name}] {s}"


def anthropic_content_text(blocks):
    """Anthropic-shaped content list -> comparable text (text else tools)."""
    texts, tools = [], []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and b.get("text"):
            texts.append(b["text"])
        elif b.get("type") == "tool_use":
            tools.append(tool_sig(b.get("name", "?"), b.get("input")))
    return "\n".join(texts) or "\n".join(tools) or None


def converse_content_text(blocks):
    """Converse-shaped content list -> comparable text."""
    texts, tools = [], []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        if b.get("text"):
            texts.append(b["text"])
        elif "toolUse" in b:
            tu = b.get("toolUse") or {}
            tools.append(tool_sig(tu.get("name", "?"), tu.get("input")))
    return "\n".join(texts) or "\n".join(tools) or None


def stream_final_text(state):
    """Streaming accumulation -> comparable text (text else tools)."""
    text = "".join(state.get("text") or [])
    if text:
        return text
    tools = [tool_sig(t.get("name", "?"), "".join(t.get("parts") or []))
             for t in state.get("tools") or []]
    return "\n".join(tools) or None
