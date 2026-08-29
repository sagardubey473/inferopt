"""Offline log ingest: analyze LLM traffic you already logged.

No proxy, no config change, nothing in your request path. Point this at
an export and it produces the same findings the live proxy would.

Canonical format - JSONL, one object per line:

    {"request": {<exact JSON body you sent>},
     "response": {<exact JSON body you got back>},
     "timestamp": 1756400000,   // optional: unix seconds or ISO-8601
     "latency_ms": 1234}        // optional

Common exporter shapes are auto-detected (Helicone `request_body` /
`response_body`, Langfuse `input`/`output`/`usage`, bare response
objects). Requests are optional but strongly recommended: without them
token accounting still works, but prefix-divergence localization - the
finding that usually matters most - cannot run.
"""
import csv
import gzip
import json
import os
import time

from . import capture, fingerprint, openai_compat, pricing

_REQ_KEYS = ("request", "request_body", "requestBody", "input", "body")
_RESP_KEYS = ("response", "response_body", "responseBody", "output",
              "completion")


def _maybe_json(v):
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None


def _pick(rec, keys):
    for k in keys:
        if k in rec:
            v = _maybe_json(rec[k])
            if v is not None:
                return v
    return None


def _parse_ts(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) / 1000.0 if v > 1e11 else float(v)  # ms vs s
    if isinstance(v, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%d %H:%M:%S"):
            try:
                import datetime
                return datetime.datetime.strptime(v, fmt).timestamp()
            except Exception:
                continue
        try:
            return float(v)
        except Exception:
            return None
    return None


def detect_rail(req, resp):
    """Which API shape is this? -> 'anthropic' | 'openai' | 'bedrock:*'."""
    if isinstance(resp, dict):
        if "choices" in resp:
            return "openai"
        if isinstance(resp.get("output"), dict) and \
                "message" in (resp.get("output") or {}):
            return "bedrock:converse"
        if isinstance(resp.get("content"), list):
            return "anthropic"
    if isinstance(req, dict):
        if "anthropic_version" in req:
            return "bedrock:invoke"
        msgs = req.get("messages")
        if isinstance(msgs, list):
            if any(isinstance(m, dict) and m.get("role") in
                   ("system", "developer") for m in msgs):
                return "openai"
            if "system" in req or "max_tokens" in req:
                return "anthropic"
        if isinstance(req.get("system"), list):
            return "bedrock:converse"
    return "openai"


def _usage_anthropic(resp):
    u = (resp or {}).get("usage") or {}
    return (u.get("input_tokens") or 0, u.get("output_tokens") or 0,
            u.get("cache_read_input_tokens") or 0,
            u.get("cache_creation_input_tokens") or 0,
            capture.anthropic_content_text(resp.get("content")))


def _usage_converse(resp):
    u = (resp or {}).get("usage") or {}
    msg = ((resp or {}).get("output") or {}).get("message") or {}
    return (u.get("inputTokens") or 0, u.get("outputTokens") or 0,
            u.get("cacheReadInputTokens") or 0,
            u.get("cacheWriteInputTokens") or 0,
            capture.converse_content_text(msg.get("content")))


def _loose_usage(rec, resp):
    """Last resort: token counts sitting at the top level of the record
    (Langfuse-style) rather than inside a provider response body."""
    u = rec.get("usage") or (resp or {}).get("usage") or {}
    if not isinstance(u, dict):
        return None
    def g(*names):
        for n in names:
            if u.get(n) is not None:
                return u[n]
        return 0
    inp = g("input", "input_tokens", "prompt_tokens", "inputTokens")
    out = g("output", "output_tokens", "completion_tokens", "outputTokens")
    if not (inp or out):
        return None
    return (inp, out, g("cache_read_input_tokens", "cached_tokens",
                        "cacheReadInputTokens"),
            g("cache_creation_input_tokens", "cacheWriteInputTokens"), None)


def normalize(rec):
    """One log record -> a row dict, or None if unusable."""
    if not isinstance(rec, dict):
        return None
    req = _pick(rec, _REQ_KEYS)
    resp = _pick(rec, _RESP_KEYS)
    if req is None and resp is None:
        # the record may itself BE a response object
        if any(k in rec for k in ("choices", "content", "usage")):
            resp = rec
        else:
            return None
    if isinstance(req, list):            # bare messages array
        req = {"messages": req}
    req = req if isinstance(req, dict) else {}

    rail = detect_rail(req, resp if isinstance(resp, dict) else None)
    resp = resp if isinstance(resp, dict) else {}

    if rail == "openai":
        i, o, cr, cw, text = openai_compat.extract_response(resp)
        fp_body = openai_compat.pseudo_body(req)
    elif rail == "bedrock:converse":
        i, o, cr, cw, text = _usage_converse(resp)
        from . import bedrock as br
        fp_body = br.pseudo_body(req, "converse")
    else:
        i, o, cr, cw, text = _usage_anthropic(resp)
        fp_body = req
    if not (i or o or cr or cw):
        loose = _loose_usage(rec, resp)
        if loose:
            i, o, cr, cw, t2 = loose
            text = text or t2

    model = (rec.get("model") or resp.get("model") or req.get("model")
             or rec.get("modelId") or "")
    ts = (_parse_ts(rec.get("timestamp") or rec.get("ts")
                    or rec.get("created_at") or rec.get("startTime"))
          or time.time())
    fp, hint = fingerprint.callsite(fp_body)
    raw = json.dumps(req)
    return {
        "ts": ts, "callsite": fp, "hint": hint, "model": model,
        "stream": 1 if req.get("stream") else 0,
        "effort": (req.get("output_config") or {}).get("effort"),
        "status": rec.get("status") or 200,
        "latency_ms": rec.get("latency_ms") or rec.get("latency") or 0,
        "input_tokens": i, "output_tokens": o,
        "cache_read_tokens": cr, "cache_write_tokens": cw,
        "cost_usd": pricing.cost_usd(model, i, o, cr, cw),
        "uses_cache_control": 1 if ('"cache_control"' in raw
                                    or '"cachePoint"' in raw) else 0,
        "prefix": fingerprint.prefix_string(fp_body),
        "body_json": raw if req else None,
        "response_text": text, "rail": rail,
    }


def read_records(path):
    """Yield raw records from JSONL, JSON array, or CSV (optionally .gz)."""
    opener = gzip.open if path.endswith(".gz") else open
    base = path[:-3] if path.endswith(".gz") else path
    ext = os.path.splitext(base)[1].lower()
    if ext == ".csv":
        with opener(path, "rt", newline="") as f:
            for row in csv.DictReader(f):
                yield row
        return
    with opener(path, "rt") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(f)
            for r in (data if isinstance(data, list) else [data]):
                yield r
            return
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def ingest(con, path, verbose=True):
    """Load a log file into the requests table. Returns (loaded, skipped)."""
    loaded = skipped = 0
    rails = {}
    for rec in read_records(path):
        row = normalize(rec)
        if row is None:
            skipped += 1
            continue
        rails[row["rail"]] = rails.get(row["rail"], 0) + 1
        con.execute(
            "INSERT INTO requests (ts,callsite,hint,model,stream,effort,"
            "status,latency_ms,input_tokens,output_tokens,cache_read_tokens,"
            "cache_write_tokens,cost_usd,uses_cache_control,prefix,body_json,"
            "response_text,rail) VALUES (:ts,:callsite,:hint,:model,:stream,"
            ":effort,:status,:latency_ms,:input_tokens,:output_tokens,"
            ":cache_read_tokens,:cache_write_tokens,:cost_usd,"
            ":uses_cache_control,:prefix,:body_json,:response_text,:rail)",
            row)
        loaded += 1
    con.commit()
    if verbose:
        shape = ", ".join(f"{k}={v}" for k, v in sorted(rails.items()))
        print(f"ingested {loaded} requests from {os.path.basename(path)}"
              + (f" ({shape})" if shape else "")
              + (f"; skipped {skipped} unrecognized" if skipped else ""))
        if loaded and not any(r["input_tokens"] for r in [row]):
            print("  note: no token counts found - check that responses "
                  "include a usage block")
    return loaded, skipped
