"""Resilience primitives for real (OpenRouter) end-to-end runs.

A long sweep is tens of thousands of API calls over hours; transient
429/5xx/timeouts, credit exhaustion, and process death are *expected*,
not exceptional. These primitives are pure and thread-safe so the
evaluator can retry, throttle, cap spend, and resume losslessly.

Nothing here imports the engine or a provider — fully unit-testable.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """Raised when the cost meter passes the hard cap. The evaluator
    catches it, finalizes from the journal, and marks the run truncated
    (a partial result is always better than a lost 6-hour run)."""


class FatalProviderError(RuntimeError):
    """A non-retryable provider failure (4xx other than 429)."""


# ── retry / backoff ────────────────────────────────────────────────────────

_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base: float = 1.0       # seconds; exponential: base * 2**(attempt-1)
    cap: float = 30.0       # max single sleep
    jitter: float = 0.1     # fraction of delay added deterministically*0

    def is_transient_status(self, status: int) -> bool:
        return status in _TRANSIENT_STATUS

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Sleep before retry `attempt` (1-based). Honors a server
        Retry-After when present and larger than our backoff."""
        backoff = min(self.cap, self.base * (2 ** max(0, attempt - 1)))
        if retry_after is not None and retry_after > 0:
            return min(self.cap, max(backoff, retry_after))
        return backoff


def retry_call(fn, policy: RetryPolicy, *, on_retry=None, sleep=time.sleep):
    """Call `fn()` with bounded exponential backoff.

    `fn` raises to signal failure; it may attach `.transient` (bool)
    and `.retry_after` (float|None) to the exception to steer policy.
    Non-transient → re-raised immediately. Exhausted attempts →
    last exception re-raised.
    """
    last: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — policy decides
            last = exc
            transient = getattr(exc, "transient", True)
            if not transient or attempt >= policy.max_attempts:
                raise
            d = policy.delay(attempt, getattr(exc, "retry_after", None))
            if on_retry is not None:
                on_retry(attempt, exc, d)
            sleep(d)
    assert last is not None
    raise last


# ── rate limiting ──────────────────────────────────────────────────────────


class RateLimiter:
    """Thread-safe minimum-interval limiter (≈ qps cap) shared across
    the concurrency pool. qps<=0 disables it."""

    def __init__(self, qps: float = 0.0):
        self._interval = 1.0 / qps if qps and qps > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self, *, now=time.monotonic, sleep=time.sleep) -> float:
        if self._interval <= 0:
            return 0.0
        with self._lock:
            t = now()
            wait = max(0.0, self._next - t)
            self._next = max(t, self._next) + self._interval
        if wait > 0:
            sleep(wait)
        return wait


# ── cost / token metering ──────────────────────────────────────────────────


@dataclass
class CostMeter:
    """Thread-safe token + USD accumulator with a hard cap.

    Pricing is per 1M tokens (OpenRouter-style). `max_usd<=0` disables
    the cap; metering still runs so the report carries spend."""

    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0
    max_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def usd(self) -> float:
        return round(
            self.prompt_tokens / 1e6 * self.price_in_per_m
            + self.completion_tokens / 1e6 * self.price_out_per_m,
            6,
        )

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.prompt_tokens += int(prompt or 0)
            self.completion_tokens += int(completion or 0)
            self.calls += 1

    def check(self) -> None:
        if self.max_usd and self.max_usd > 0 and self.usd >= self.max_usd:
            raise BudgetExceeded(
                f"spend ${self.usd:.4f} ≥ cap ${self.max_usd:.2f} "
                f"({self.calls} calls, {self.prompt_tokens}+"
                f"{self.completion_tokens} tok)"
            )

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd": self.usd,
            "max_usd": self.max_usd,
        }


# ── checkpoint / resume journal ────────────────────────────────────────────


def episode_key(pack: str, level: str, split: str, seed: int) -> str:
    return f"{pack}|{level}|{split}|{seed}"


class RunJournal:
    """Append-only JSONL of completed episodes. Resume = skip keys
    already present; the aggregate is rebuilt from the journal so a
    killed run continues losslessly."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def done_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        keys: set[str] = set()
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["_key"])
            except Exception:  # noqa: BLE001 — tolerate a torn last line
                continue
        return keys

    def append(self, key: str, record: dict) -> None:
        row = dict(record)
        row["_key"] = key
        line = json.dumps(row)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
        return out
