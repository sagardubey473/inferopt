"""OpenRouter model catalog: live pricing for ~400 models, cached locally.

Hardcoding prices for every OpenAI-compatible model is hopeless, so the
openai rail looks them up from OpenRouter's public catalog (no auth) and
caches the result for a day. Prices there are per-token strings; we
normalize to $/MTok to match inferopt.pricing.
"""
import json
import os
import time
import urllib.request

CATALOG_URL = "https://openrouter.ai/api/v1/models"
CACHE_PATH = os.path.expanduser("~/.inferopt/openrouter-models.json")
TTL = 86400
_mem = None


def _load_cache():
    try:
        with open(CACHE_PATH) as f:
            blob = json.load(f)
        if time.time() - blob.get("fetched_at", 0) < TTL:
            return blob["models"]
    except Exception:
        pass
    return None


def _fetch():
    req = urllib.request.Request(CATALOG_URL,
                                 headers={"User-Agent": "inferopt"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    models = {}
    for m in data.get("data", []):
        p = m.get("pricing") or {}

        def rate(key):
            try:
                v = float(p.get(key, 0) or 0)
            except (TypeError, ValueError):
                return None
            return None if v < 0 else v * 1e6      # $/token -> $/MTok

        models[m["id"]] = {
            "in": rate("prompt"), "out": rate("completion"),
            "cache_read": rate("input_cache_read"),
            "cache_write": rate("input_cache_write"),
            "name": m.get("name", m["id"]),
        }
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"fetched_at": time.time(), "models": models}, f)
    return models


def models(refresh=False):
    global _mem
    if _mem is not None and not refresh:
        return _mem
    _mem = (None if refresh else _load_cache())
    if _mem is None:
        try:
            _mem = _fetch()
        except Exception:
            _mem = {}
    return _mem


def lookup(model_id):
    """Return dict with in/out/cache_read/cache_write in $/MTok, or None."""
    if not model_id:
        return None
    cat = models()
    if model_id in cat:
        return cat[model_id]
    # tolerate ":free" / ":nitro" style suffixes and bare vendor ids
    base = model_id.split(":")[0]
    if base in cat:
        return cat[base]
    for k in cat:
        if k.split(":")[0] == base:
            return cat[k]
    return None
