"""Transparent logging proxy for the Anthropic API.

Point any Anthropic SDK at it with:  export ANTHROPIC_BASE_URL=http://127.0.0.1:8484
Requests are forwarded byte-for-byte to api.anthropic.com (streaming included);
/v1/messages traffic is instrumented into a local SQLite DB. Nothing leaves
your machine except the original API call.
"""
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from . import db, fingerprint, pricing

UPSTREAM = os.environ.get("INFEROPT_UPSTREAM", "https://api.anthropic.com")
STORE_BODIES = os.environ.get("INFEROPT_STORE_BODIES", "1") != "0"

# hop-by-hop / recomputed headers we never forward
_REQ_DROP = {"host", "content-length", "connection", "accept-encoding",
             "transfer-encoding"}
_RESP_DROP = {"content-length", "content-encoding", "transfer-encoding",
              "connection"}

_state = {"con": None, "client": None}


@asynccontextmanager
async def _lifespan(app):
    _state["con"] = db.connect()
    _state["client"] = httpx.AsyncClient(
        base_url=UPSTREAM, timeout=httpx.Timeout(600.0, connect=15.0)
    )
    yield
    await _state["client"].aclose()
    _state["con"].close()


app = FastAPI(lifespan=_lifespan)


def _insert(body, raw, status, model, input_t, output_t, cr, cw, latency,
            resp_text):
    fp, hint = fingerprint.callsite(body)
    cost = pricing.cost_usd(model, input_t, output_t, cr, cw)
    effort = (body.get("output_config") or {}).get("effort")
    _state["con"].execute(
        "INSERT INTO requests (ts,callsite,hint,model,stream,effort,status,"
        "latency_ms,input_tokens,output_tokens,cache_read_tokens,"
        "cache_write_tokens,cost_usd,uses_cache_control,prefix,body_json,"
        "response_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            time.time(), fp, hint, model,
            1 if body.get("stream") else 0, effort, status, latency,
            input_t, output_t, cr, cw, cost,
            1 if b'"cache_control"' in raw else 0,
            fingerprint.prefix_string(body),
            raw.decode("utf-8", "replace") if STORE_BODIES else None,
            resp_text,
        ),
    )
    _state["con"].commit()
    cost_s = f"${cost:.4f}" if cost is not None else "$?"
    print(f"[inferopt] {model or '?'} site={fp} in={input_t} out={output_t} "
          f"cache_read={cr} cache_write={cw} {cost_s} {latency:.0f}ms",
          flush=True)


def _usage_fields(usage):
    usage = usage or {}
    return (
        usage.get("input_tokens") or 0,
        usage.get("output_tokens") or 0,
        usage.get("cache_read_input_tokens") or 0,
        usage.get("cache_creation_input_tokens") or 0,
    )


def _record_json(body, raw, r, latency):
    model = body.get("model")
    input_t = output_t = cr = cw = 0
    resp_text = None
    try:
        data = r.json()
        model = data.get("model") or model
        input_t, output_t, cr, cw = _usage_fields(data.get("usage"))
        resp_text = "\n".join(
            b.get("text", "") for b in data.get("content") or []
            if isinstance(b, dict) and b.get("type") == "text"
        ) or None
    except Exception:
        pass
    _insert(body, raw, r.status_code, model, input_t, output_t, cr, cw,
            latency, resp_text)


def _parse_sse_line(line, state):
    line = line.strip()
    if not line.startswith(b"data:"):
        return
    try:
        evt = json.loads(line[5:].strip())
    except Exception:
        return
    t = evt.get("type")
    if t == "message_start":
        msg = evt.get("message") or {}
        state["model"] = msg.get("model") or state["model"]
        i, o, cr, cw = _usage_fields(msg.get("usage"))
        state.update(input=i, output=o, cr=cr, cw=cw)
    elif t == "message_delta":
        u = evt.get("usage") or {}
        if u.get("output_tokens") is not None:
            state["output"] = u["output_tokens"]  # cumulative
    elif t == "content_block_delta":
        d = evt.get("delta") or {}
        if d.get("type") == "text_delta":
            state["text"].append(d.get("text", ""))


async def _relay_stream(url, headers, raw, body):
    t0 = time.time()
    client = _state["client"]
    req = client.build_request("POST", url, headers=headers, content=raw)
    upstream = await client.send(req, stream=True)
    state = {"input": 0, "output": 0, "cr": 0, "cw": 0,
             "model": body.get("model"), "text": []}

    async def gen():
        buf = b""
        try:
            async for chunk in upstream.aiter_raw():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    _parse_sse_line(line, state)
                yield chunk
        finally:
            await upstream.aclose()
            _insert(body, raw, upstream.status_code, state["model"],
                    state["input"], state["output"], state["cr"], state["cw"],
                    (time.time() - t0) * 1000,
                    "".join(state["text"]) or None)

    resp_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() not in _RESP_DROP}
    return StreamingResponse(gen(), status_code=upstream.status_code,
                             headers=resp_headers)


@app.api_route("/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def relay(request: Request, path: str):
    raw = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _REQ_DROP}
    url = "/" + path
    if request.url.query:
        url += "?" + request.url.query

    body = None
    instrument = path == "v1/messages" and request.method == "POST"
    if instrument:
        try:
            body = json.loads(raw)
        except Exception:
            instrument = False

    if instrument and body.get("stream"):
        return await _relay_stream(url, headers, raw, body)

    t0 = time.time()
    r = await _state["client"].request(
        request.method, url, headers=headers, content=raw
    )
    latency = (time.time() - t0) * 1000
    if instrument:
        _record_json(body, raw, r, latency)
    resp_headers = {k: v for k, v in r.headers.items()
                    if k.lower() not in _RESP_DROP}
    return Response(content=r.content, status_code=r.status_code,
                    headers=resp_headers)


def run(port=8484):
    import uvicorn
    print(f"[inferopt] proxy on http://127.0.0.1:{port}  ->  {UPSTREAM}")
    print(f"[inferopt] logging to {db.DEFAULT_DB}"
          + ("" if STORE_BODIES else "  (metadata only, bodies not stored)"))
    print(f"[inferopt] point your code at it with:")
    print(f"[inferopt]   export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
