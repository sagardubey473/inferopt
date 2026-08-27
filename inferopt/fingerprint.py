"""Call-site fingerprinting: cluster structurally similar requests.

A "call site" is a group of requests that share the same tool set and the
same *normalized* system prompt (volatile substrings like timestamps, UUIDs
and numbers replaced with placeholders). It is the unit of analysis for
every finding the report produces.
"""
import hashlib
import json
import re

_PATTERNS = [
    (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"), "<uuid>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?"
                r"(Z|[+-]\d{2}:?\d{2})?"), "<ts>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
    (re.compile(r"\b[0-9a-fA-F]{16,}\b"), "<hex>"),
    (re.compile(r"\d+"), "<n>"),
]


def normalize(text):
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def _system_text(body):
    sys = body.get("system") or ""
    if isinstance(sys, list):
        sys = " ".join(b.get("text", "") for b in sys if isinstance(b, dict))
    return sys if isinstance(sys, str) else json.dumps(sys)


def _first_user_text(body):
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return normalize(c)
            if isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        return normalize(blk.get("text", ""))
    return ""


def callsite(body):
    """Return (fingerprint, human-readable hint)."""
    tools = sorted(
        t.get("name", t.get("type", "?")) for t in body.get("tools") or []
    )
    sys_norm = normalize(_system_text(body))
    key = json.dumps({"tools": tools, "system": sys_norm})
    fp = hashlib.sha1(key.encode()).hexdigest()[:12]
    hint = (sys_norm.strip() or _first_user_text(body).strip())[:80]
    return fp, hint.replace("\n", " ") or "(no system prompt)"


def prefix_string(body):
    """Serialized cacheable prefix (tools -> system), matching API render order.

    Used for byte-level divergence detection between requests at the same
    call site: the first differing byte position IS the cache invalidator.
    """
    return json.dumps(
        {"tools": body.get("tools"), "system": body.get("system")},
        ensure_ascii=False,
    )
