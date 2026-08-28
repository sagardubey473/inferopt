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

from . import capture, db, fingerprint, pricing

UPSTREAM = os.environ.get("INFEROPT_UPSTREAM", "https://api.anthropic.com")
STORE_BODIES = os.environ.get("INFEROPT_STORE_BODIES", "1") != "0"
BEDROCK_REGION = (os.environ.get("INFEROPT_BEDROCK_REGION")
                  or os.environ.get("AWS_REGION")
                  or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
BEDROCK_UPSTREAM = os.environ.get(
    "INFEROPT_BEDROCK_UPSTREAM",
    f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com")
_BEDROCK_ACTIONS = {"invoke", "converse",
                    "invoke-with-response-stream", "converse-stream"}
OPENAI_UPSTREAM = os.environ.get("INFEROPT_OPENAI_UPSTREAM",
                                 "https://openrouter.ai/api")

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
            resp_text, stream=None, rail="anthropic", fp_body=None):
    fp_body = fp_body if fp_body is not None else body
    fp, hint = fingerprint.callsite(fp_body)
    cost = pricing.cost_usd(model, input_t, output_t, cr, cw)
    effort = (body.get("output_config") or {}).get("effort")
    if stream is None:
        stream = 1 if body.get("stream") else 0
    _state["con"].execute(
        "INSERT INTO requests (ts,callsite,hint,model,stream,effort,status,"
        "latency_ms,input_tokens,output_tokens,cache_read_tokens,"
        "cache_write_tokens,cost_usd,uses_cache_control,prefix,body_json,"
        "response_text,rail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            time.time(), fp, hint, model, stream, effort, status, latency,
            input_t, output_t, cr, cw, cost,
            1 if (b'"cache_control"' in raw or b'"cachePoint"' in raw) else 0,
            fingerprint.prefix_string(fp_body),
            raw.decode("utf-8", "replace") if STORE_BODIES else None,
            resp_text, rail,
        ),
    )
    _state["con"].commit()
    cost_s = f"${cost:.4f}" if cost is not None else "$?"
    print(f"[inferopt] [{rail}] {model or '?'} site={fp} in={input_t} "
          f"out={output_t} cache_read={cr} cache_write={cw} {cost_s} "
          f"{latency:.0f}ms", flush=True)


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
        resp_text = capture.anthropic_content_text(data.get("content"))
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


async def _relay_stream(url, headers, raw, body):
    t0 = time.time()
    client = _state["client"]
    req = client.build_request("POST", url, headers=headers, content=raw)
    upstream = await client.send(req, stream=True)
    state = {"input": 0, "output": 0, "cr": 0, "cw": 0,
             "model": body.get("model"), "text": [], "tools": []}

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
                    capture.stream_final_text(state))

    resp_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() not in _RESP_DROP}
    return StreamingResponse(gen(), status_code=upstream.status_code,
                             headers=resp_headers)


async def _relay_openai(request, path, raw):
    """OpenAI-compatible rail: forward verbatim (auth is a bearer header,
    nothing to re-sign) and instrument chat completions."""
    from . import openai_compat as oc
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _REQ_DROP}
    url = OPENAI_UPSTREAM + "/" + path
    if request.url.query:
        url += "?" + request.url.query

    instrument = path.rstrip("/").endswith("chat/completions")
    body = {}
    if instrument:
        try:
            body = json.loads(raw)
        except Exception:
            instrument = False
    fp_body = oc.pseudo_body(body) if instrument else {}
    client = _state["client"]
    t0 = time.time()

    if instrument and body.get("stream"):
        req = client.build_request("POST", url, headers=headers, content=raw)
        upstream = await client.send(req, stream=True)
        state = {"input": 0, "output": 0, "cr": 0, "cw": 0, "text": [],
                 "tools": []}

        async def gen():
            buf = b""
            try:
                async for chunk in upstream.aiter_raw():
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        oc.apply_stream_event(line, state)
                    yield chunk
            finally:
                await upstream.aclose()
                _insert(body, raw, upstream.status_code, body.get("model"),
                        state["input"], state["output"], state["cr"],
                        state["cw"], (time.time() - t0) * 1000,
                        capture.stream_final_text(state),
                        stream=1, rail="openai", fp_body=fp_body)

        resp_headers = {k: v for k, v in upstream.headers.items()
                        if k.lower() not in _RESP_DROP}
        return StreamingResponse(gen(), status_code=upstream.status_code,
                                 headers=resp_headers)

    r = await client.request(request.method, url, headers=headers,
                             content=raw)
    latency = (time.time() - t0) * 1000
    if instrument:
        i = o = cr = cw = 0
        text = None
        model = body.get("model")
        try:
            data = r.json()
            model = data.get("model") or model
            i, o, cr, cw, text = oc.extract_response(data)
        except Exception:
            pass
        _insert(body, raw, r.status_code, model, i, o, cr, cw, latency, text,
                stream=0, rail="openai", fp_body=fp_body)
    resp_headers = {k: v for k, v in r.headers.items()
                    if k.lower() not in _RESP_DROP}
    return Response(content=r.content, status_code=r.status_code,
                    headers=resp_headers)


async def _relay_bedrock(request, path, raw):
    from . import bedrock as br
    parts = path.split("/")
    model_id = ""
    action = ""
    if len(parts) >= 3 and parts[0] == "model":
        import urllib.parse
        model_id = urllib.parse.unquote(parts[1])
        action = parts[2]
    raw_path = request.scope.get("raw_path", b"/" + path.encode()).decode()
    url = BEDROCK_UPSTREAM + raw_path
    fwd = {}
    for k in ("content-type", "accept"):
        v = request.headers.get(k)
        if v:
            fwd[k] = v
    try:
        signed = br.sign(request.method, url, fwd, raw, BEDROCK_REGION)
    except Exception as e:
        return Response(
            content=json.dumps({"inferopt_error": str(e)}),
            status_code=500, media_type="application/json")

    instrument = action in _BEDROCK_ACTIONS
    body = {}
    if instrument:
        try:
            body = json.loads(raw)
        except Exception:
            instrument = False
    fp_body = br.pseudo_body(body, action) if instrument else {}
    rail = f"bedrock:{action.split('-')[0]}" if instrument else "bedrock"
    streaming = action.endswith("stream")
    client = _state["client"]
    t0 = time.time()

    if streaming:
        req = client.build_request(request.method, url, headers=signed,
                                   content=raw)
        upstream = await client.send(req, stream=True)
        es = br.EventStream()
        state = {"input": 0, "output": 0, "cr": 0, "cw": 0, "text": [],
                 "tools": []}

        async def gen():
            try:
                async for chunk in upstream.aiter_raw():
                    for hdrs, payload in es.feed(chunk):
                        br.apply_stream_event(action, hdrs, payload, state)
                    yield chunk
            finally:
                await upstream.aclose()
                if instrument:
                    _insert(body, raw, upstream.status_code, model_id,
                            state["input"], state["output"], state["cr"],
                            state["cw"], (time.time() - t0) * 1000,
                            capture.stream_final_text(state),
                            stream=1, rail=rail, fp_body=fp_body)

        resp_headers = {k: v for k, v in upstream.headers.items()
                        if k.lower() not in _RESP_DROP}
        return StreamingResponse(gen(), status_code=upstream.status_code,
                                 headers=resp_headers)

    r = await client.request(request.method, url, headers=signed, content=raw)
    latency = (time.time() - t0) * 1000
    if instrument:
        i = o = cr = cw = 0
        text = None
        try:
            i, o, cr, cw, text = br.extract_response(action, r.json())
        except Exception:
            pass
        _insert(body, raw, r.status_code, model_id, i, o, cr, cw, latency,
                text, stream=0, rail=rail, fp_body=fp_body)
    resp_headers = {k: v for k, v in r.headers.items()
                    if k.lower() not in _RESP_DROP}
    return Response(content=r.content, status_code=r.status_code,
                    headers=resp_headers)


@app.api_route("/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def relay(request: Request, path: str):
    raw = await request.body()
    if path.startswith(("model/", "async-invoke", "guardrail")):
        return await _relay_bedrock(request, path, raw)
    if "chat/completions" in path or path.startswith(("v1/models",
                                                      "api/v1/")):
        return await _relay_openai(request, path, raw)
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
    print(f"[inferopt] proxy on http://127.0.0.1:{port}")
    print(f"[inferopt]   anthropic rail -> {UPSTREAM}")
    print(f"[inferopt]   bedrock rail   -> {BEDROCK_UPSTREAM} "
          f"(region {BEDROCK_REGION}, re-signed with local AWS creds)")
    print(f"[inferopt]   openai rail    -> {OPENAI_UPSTREAM} "
          f"(/v1/chat/completions; pricing from the OpenRouter catalog)")
    print(f"[inferopt] logging to {db.DEFAULT_DB}"
          + ("" if STORE_BODIES else "  (metadata only, bodies not stored)"))
    print(f"[inferopt] point your code at it with:")
    print(f"[inferopt]   export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}")
    print(f"[inferopt]   export AWS_ENDPOINT_URL_BEDROCK_RUNTIME=http://127.0.0.1:{port}")
    print(f"[inferopt]   OpenAI-compatible base url: "
          f"http://127.0.0.1:{port}/v1")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
