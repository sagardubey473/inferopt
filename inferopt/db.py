import os
import sqlite3

DEFAULT_DB = os.environ.get(
    "INFEROPT_DB", os.path.expanduser("~/.inferopt/inferopt.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, callsite TEXT, hint TEXT, model TEXT,
  stream INTEGER, effort TEXT, status INTEGER, latency_ms REAL,
  input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
  cache_read_tokens INTEGER DEFAULT 0, cache_write_tokens INTEGER DEFAULT 0,
  cost_usd REAL, uses_cache_control INTEGER, prefix TEXT,
  rail TEXT DEFAULT 'anthropic',
  body_json TEXT, response_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_callsite ON requests(callsite);
CREATE INDEX IF NOT EXISTS idx_ts ON requests(ts);
CREATE TABLE IF NOT EXISTS validations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, callsite TEXT, alt_model TEXT, n INTEGER,
  equivalent INTEGER DEFAULT 0, better INTEGER DEFAULT 0,
  worse INTEGER DEFAULT 0, mismatches INTEGER DEFAULT 0,
  decision TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS idx_val ON validations(callsite, alt_model);
"""


def connect(path=None):
    path = path or DEFAULT_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    try:  # migrate DBs created before the Bedrock rail existed
        con.execute("ALTER TABLE requests ADD COLUMN rail TEXT DEFAULT 'anthropic'")
    except sqlite3.OperationalError:
        pass
    con.row_factory = sqlite3.Row
    return con
