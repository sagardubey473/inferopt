"""Anthropic first-party API pricing, USD per million tokens.

Verified against platform.claude.com/docs/en/about-claude/pricing on 2026-08-27.
Cache read = 0.1x base input; 5-minute cache write = 1.25x; batch = 50% off
input AND output (multipliers stack).
"""

PRICES = {  # model-id prefix -> (input $/MTok, output $/MTok)
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # legacy (still common on Bedrock)
    "claude-opus-4-1": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.8, 4.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-haiku": (0.25, 1.25),
}

CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = 1.25   # 5-minute ephemeral; 1-hour TTL is 2.0x
BATCH_MULT = 0.50

# Cheaper same-API tiers worth testing via `inferopt replay`
CHEAPER_TIERS = ["claude-sonnet-5", "claude-haiku-4-5"]

# Models that reject temperature/top_p/top_k (removed sampling params)
NO_SAMPLING = {
    "claude-fable-5", "claude-mythos-5", "claude-opus-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5",
}


def catalog_rates(model):
    """Full rate card from the OpenRouter catalog (openai rail), $/MTok."""
    try:
        from . import catalog
        return catalog.lookup(model)
    except Exception:
        return None


def resolve(model):
    """Map a request model string to a pricing key. Handles first-party ids
    (date-suffixed or not) and Bedrock ids like
    'us.anthropic.claude-sonnet-5-v1:0'."""
    if not model:
        return None
    for pre in ("us.", "eu.", "apac.", "global.", "jp."):
        if model.startswith(pre):
            model = model[len(pre):]
            break
    if model.startswith("anthropic."):
        model = model[len("anthropic."):]
    if model in PRICES:
        return model
    best = None
    for key in PRICES:
        if model.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return best


def rates(model):
    key = resolve(model)
    r = PRICES.get(key) if key else None
    if r is not None:
        return r
    c = catalog_rates(model)
    if c and c.get("in") is not None:
        return (c["in"], c.get("out") or 0.0)
    return None


def is_first_party(model):
    """True when the id maps to a hardcoded first-party/Bedrock price."""
    key = resolve(model)
    return bool(key and key in PRICES)


def cost_usd(model, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0):
    r = rates(model)
    if r is not None:
        inp, out = r
        return (
            input_tokens * inp
            + output_tokens * out
            + cache_read * inp * CACHE_READ_MULT
            + cache_write * inp * CACHE_WRITE_MULT
        ) / 1e6
    c = catalog_rates(model)
    if c is None or c.get("in") is None:
        return None
    inp, out = c["in"], c["out"] or 0.0
    cr = c.get("cache_read")
    cw = c.get("cache_write")
    cr = inp * CACHE_READ_MULT if cr is None else cr
    cw = inp * CACHE_WRITE_MULT if cw is None else cw
    return (input_tokens * inp + output_tokens * out
            + cache_read * cr + cache_write * cw) / 1e6
