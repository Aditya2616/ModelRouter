"""SQLite cost ledger. Stores a hash of the prompt, never the prompt text or any credential."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass


@dataclass
class LedgerRow:
    prompt_hash: str
    difficulty: float
    tier: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float          # what the routed request cost (0 for a cache hit)
    big_only_usd: float      # what the same request would have cost on the big tier alone
    escalated: bool
    cached: bool
    latency_s: float
    cache_saved_usd: float = 0.0


SCHEMA = """CREATE TABLE IF NOT EXISTS ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, prompt_hash TEXT, difficulty REAL, tier TEXT, model TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL, big_only_usd REAL, escalated INTEGER,
  cached INTEGER, latency_s REAL, cache_saved_usd REAL)"""


class Ledger:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(SCHEMA)
            self._db.commit()

    def add(self, r: LedgerRow) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO ledger (ts,prompt_hash,difficulty,tier,model,prompt_tokens,completion_tokens,cost_usd,"
                "big_only_usd,escalated,cached,latency_s,cache_saved_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), r.prompt_hash, r.difficulty, r.tier, r.model, r.prompt_tokens, r.completion_tokens,
                 r.cost_usd, r.big_only_usd, int(r.escalated), int(r.cached), r.latency_s, r.cache_saved_usd))
            self._db.commit()

    def summary(self) -> dict:
        """Routing saving is computed on non-cached requests only, so caching is not credited to the router."""
        with self._lock:
            c = self._db.cursor()
            n, = c.execute("SELECT COUNT(*) FROM ledger").fetchone()
            row = c.execute("SELECT COALESCE(SUM(cost_usd),0), COALESCE(SUM(big_only_usd),0), COUNT(*), "
                            "COALESCE(SUM(tier='big'),0), COALESCE(SUM(escalated),0) FROM ledger WHERE cached=0").fetchone()
            cache_n, cache_saved = c.execute("SELECT COUNT(*), COALESCE(SUM(cache_saved_usd),0) FROM ledger "
                                             "WHERE cached=1").fetchone()
        cost, big_only, live, big_n, esc = row
        return {
            "requests": n, "live_requests": live, "cache_hits": cache_n,
            "routed_cost_usd": cost, "big_only_cost_usd": big_only,
            "routing_saving_usd": big_only - cost,
            "routing_saving_pct": (100 * (big_only - cost) / big_only) if big_only > 0 else 0.0,
            "big_share": (big_n / live) if live else 0.0, "escalations": esc, "cache_saved_usd": cache_saved,
        }

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            cur = self._db.execute("SELECT ts,prompt_hash,difficulty,tier,model,prompt_tokens,completion_tokens,cost_usd,"
                                   "big_only_usd,escalated,cached,latency_s FROM ledger ORDER BY id DESC LIMIT ?", (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
