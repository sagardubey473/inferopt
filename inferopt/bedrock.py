"""Bedrock rail: SigV4 re-signing, AWS eventstream parsing, shape adapters.

Bedrock requests are SigV4-signed over the Host header, so a transparent
proxy invalidates them. Instead the app points at the proxy via
AWS_ENDPOINT_URL_BEDROCK_RUNTIME and the proxy re-signs with the local AWS
credential chain (env / profile / SSO) before forwarding.
"""
import base64
import json
import struct

from . import capture

_session = None


def sign(method, url, headers, body, region):
    """Re-sign a request with the local AWS credential chain."""
    global _session
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    import botocore.session
    if _session is None:
        _session = botocore.session.Session()
    creds = _session.get_credentials()
    if creds is None:
        raise RuntimeError(
            "no AWS credentials found (env/profile/SSO) - the proxy needs "
            "them to re-sign Bedrock requests")
    req = AWSRequest(method=method, url=url, data=body, headers=dict(headers))
    SigV4Auth(creds.get_frozen_credentials(), "bedrock", region).add_auth(req)
    return dict(req.headers)


class EventStream:
    """Minimal parser for application/vnd.amazon.eventstream framing."""

    def __init__(self):
        self.buf = b""

    def feed(self, chunk):
        self.buf += chunk
        out = []
        while len(self.buf) >= 12:
            total = struct.unpack(">I", self.buf[:4])[0]
            if total < 16 or len(self.buf) < total:
                break
            hlen = struct.unpack(">I", self.buf[4:8])[0]
            frame = self.buf[:total]
            self.buf = self.buf[total:]
            out.append((self._headers(frame[12:12 + hlen]),
                        frame[12 + hlen:total - 4]))
        return out

    @staticmethod
    def _headers(b):
        h, i = {}, 0
        try:
            while i < len(b):
                nlen = b[i]; i += 1
                name = b[i:i + nlen].decode("utf-8", "replace"); i += nlen
                vtype = b[i]; i += 1
                if vtype == 7:  # string
                    vlen = struct.unpack(">H", b[i:i + 2])[0]; i += 2
                    h[name] = b[i:i + vlen].decode("utf-8", "replace")
                    i += vlen
                elif vtype in (0, 1):
                    h[name] = (vtype == 0)
                elif vtype == 2:
                    h[name] = b[i]; i += 1
                elif vtype == 3:
                    h[name] = struct.unpack(">h", b[i:i + 2])[0]; i += 2
                elif vtype == 4:
                    h[name] = struct.unpack(">i", b[i:i + 4])[0]; i += 4
                elif vtype in (5, 8):
                    h[name] = struct.unpack(">q", b[i:i + 8])[0]; i += 8
                elif vtype == 6:  # bytes
                    vlen = struct.unpack(">H", b[i:i + 2])[0]; i += 2
                    h[name] = b[i:i + vlen]; i += vlen
                elif vtype == 9:  # uuid
                    h[name] = b[i:i + 16].hex(); i += 16
                else:
                    break
        except Exception:
            pass
        return h


def pseudo_body(body, action):
    """Adapt a Bedrock request body to the Anthropic-Messages-ish shape the
    fingerprint/prefix code expects. invoke bodies already ARE that shape;
    converse bodies get translated."""
    if action.startswith("invoke"):
        return body
    # converse / converse-stream
    system = " ".join(s.get("text", "") for s in body.get("system") or []
                      if isinstance(s, dict))
    msgs = []
    for m in body.get("messages") or []:
        content = [{"type": "text", "text": b.get("text", "")}
                   for b in m.get("content") or []
                   if isinstance(b, dict) and "text" in b]
        msgs.append({"role": m.get("role"), "content": content})
    tools = []
    tc = body.get("toolConfig") or {}
    for t in tc.get("tools") or []:
        spec = t.get("toolSpec") or {}
        if spec.get("name"):
            tools.append({"name": spec["name"]})
    out = {"system": system, "messages": msgs}
    if tools:
        out["tools"] = tools
    return out


def extract_response(action, data):
    """(input, output, cache_read, cache_write, text) from a non-stream
    Bedrock response body."""
    if action == "invoke":
        u = data.get("usage") or {}
        text = capture.anthropic_content_text(data.get("content"))
        return (u.get("input_tokens") or 0, u.get("output_tokens") or 0,
                u.get("cache_read_input_tokens") or 0,
                u.get("cache_creation_input_tokens") or 0, text)
    # converse
    u = data.get("usage") or {}
    msg = (data.get("output") or {}).get("message") or {}
    text = capture.converse_content_text(msg.get("content"))
    return (u.get("inputTokens") or 0, u.get("outputTokens") or 0,
            u.get("cacheReadInputTokens") or 0,
            u.get("cacheWriteInputTokens") or 0, text)


def apply_stream_event(action, headers, payload, state):
    """Update the usage-accumulation state from one eventstream frame."""
    try:
        evt = json.loads(payload)
    except Exception:
        return
    if action == "invoke-with-response-stream":
        # chunk payloads wrap the Anthropic SSE event in base64 "bytes"
        if "bytes" in evt:
            try:
                evt = json.loads(base64.b64decode(evt["bytes"]))
            except Exception:
                return
        t = evt.get("type")
        if t == "message_start":
            u = (evt.get("message") or {}).get("usage") or {}
            state["input"] = u.get("input_tokens") or 0
            state["cr"] = u.get("cache_read_input_tokens") or 0
            state["cw"] = u.get("cache_creation_input_tokens") or 0
            state["output"] = u.get("output_tokens") or 0
        elif t == "message_delta":
            u = evt.get("usage") or {}
            if u.get("output_tokens") is not None:
                state["output"] = u["output_tokens"]
        elif t == "content_block_start":
            cb = evt.get("content_block") or {}
            if cb.get("type") == "tool_use":
                state.setdefault("tools", []).append(
                    {"name": cb.get("name", "?"), "parts": []})
        elif t == "content_block_delta":
            d = evt.get("delta") or {}
            if d.get("type") == "text_delta":
                state["text"].append(d.get("text", ""))
            elif d.get("type") == "input_json_delta" and state.get("tools"):
                state["tools"][-1]["parts"].append(d.get("partial_json", ""))
        return
    # converse-stream: event type lives in the frame header
    et = headers.get(":event-type")
    if et == "contentBlockStart":
        tu = (evt.get("start") or {}).get("toolUse")
        if tu:
            state.setdefault("tools", []).append(
                {"name": tu.get("name", "?"), "parts": []})
    elif et == "contentBlockDelta":
        d = evt.get("delta") or {}
        if "text" in d:
            state["text"].append(d.get("text", ""))
        elif "toolUse" in d and state.get("tools"):
            state["tools"][-1]["parts"].append(
                (d.get("toolUse") or {}).get("input", ""))
    elif et == "metadata":
        u = evt.get("usage") or {}
        state["input"] = u.get("inputTokens") or 0
        state["output"] = u.get("outputTokens") or 0
        state["cr"] = u.get("cacheReadInputTokens") or 0
        state["cw"] = u.get("cacheWriteInputTokens") or 0
