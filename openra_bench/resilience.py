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


def episode_key(pack: str, level: str, split: str, seed: int,
                fog_mode: str = "vision", repeat: int = 0) -> str:
    """Stable key for the run journal —
    pack|level|split|seed|fog_mode  (repeat appended when > 0).

    fog_mode is included so a (pack, level, split, seed) eval'd in
    `vision` and again in `structured` are treated as distinct cells;
    resume on the same out_dir won't accidentally skip a new modality.

    `repeat` is appended ONLY when > 0 so:
      - back-compat: existing journals (no repeat suffix) match new
        `repeat=0` keys exactly,
      - pass^N stability sweeps: rep=1, rep=2 each carry a unique key
        so the per-process dedupe doesn't trip and the resume-gate
        counts each attempt distinctly."""
    base = f"{pack}|{level}|{split}|{seed}|{fog_mode}"
    return f"{base}|rep{repeat}" if repeat else base


class DuplicateJournalKey(RuntimeError):
    """Raised when the same `_key` is appended twice within a single
    process. v1.0 Qwen 9B sweep had `adversarial-duel:easy` show up
    twice in the journal; this hard-stops that class of footgun."""


class JournalRunIdMismatch(RuntimeError):
    """Raised when a journal's header `run_id` does not match the
    current process. Acknowledge by passing `--ignore-run-id` in the
    CLI (or `ignore_run_id=True` on RunJournal) so the operator has
    to consciously merge two sweep runs into the same journal."""


class RunJournal:
    """Append-only JSONL of completed episodes. Resume = skip keys
    already present; the aggregate is rebuilt from the journal so a
    killed run continues losslessly.

    Hardened for production multi-hour sweeps (v11 audit fixes):

    * Header (`{"_meta": true, ...}`) is written once on first append;
      includes `run_id`, `model`, `code_version` (git SHA when
      available) so a re-open can verify it's the SAME run that's
      being resumed. Mismatch → `JournalRunIdMismatch` unless
      `ignore_run_id=True` (the explicit "merge two runs" knob).

    * In-memory `_key` dedupe inside the append lock: a second
      append of the same key in the same process raises
      `DuplicateJournalKey`. This catches the v1.0 dup-key footgun
      that produced 205/653 journal↔disk mismatches. The seen set
      is seeded from existing (non-meta) records on construction so
      a resume can't re-add a key the prior process already wrote.
    """

    def __init__(self, path: str | Path, *, run_id: str | None = None,
                 model: str | None = None, code_version: str | None = None,
                 ignore_run_id: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.run_id = run_id
        self.model = model
        self.code_version = code_version
        self.ignore_run_id = ignore_run_id
        # Seed the in-memory dedupe set from existing rows so a resume
        # can't re-append a key the prior process already wrote.
        #
        # FOOTGUN HISTORY: errored rows used to be added here, which
        # crashed the launcher when the resume gate (correctly) retried
        # them — `done_keys()` excluded the error row from the done set,
        # so the task got re-submitted; on completion `append()` then
        # raised `DuplicateJournalKey` because `_seen_keys` still held
        # the error row's key. The fix mirrors `done_keys()`: errored
        # rows are NOT added to the dedupe set, so a retry's append is
        # allowed through. Downstream readers must dedup by `_key`
        # (`done_keys()` already does — uses a set + filters errors).
        self._seen_keys: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if rec.get("_meta"):
                    self._verify_header(rec)
                    continue
                if rec.get("outcome") == "error":
                    continue  # retry must be allowed to append
                k = rec.get("_key")
                if k is not None:
                    self._seen_keys.add(k)

    # ── header ────────────────────────────────────────────────────────

    def _verify_header(self, meta: dict) -> None:
        """If `run_id` is set on this RunJournal, the on-disk header
        must match (unless the caller passed `ignore_run_id=True`).
        A v1.0 footgun: two parallel sweep processes pointed at the
        same journal silently merged into a frankenstein run; this
        forces the operator to acknowledge it."""
        if not self.run_id or self.ignore_run_id:
            return
        on_disk = meta.get("run_id")
        if on_disk and on_disk != self.run_id:
            raise JournalRunIdMismatch(
                f"journal {self.path} was written by run_id={on_disk!r} "
                f"but the current process is run_id={self.run_id!r}. "
                f"Pass --ignore-run-id to merge runs explicitly."
            )

    def _ensure_header(self) -> None:
        """Write the `_meta` header line on first append. Idempotent —
        a file that already starts with `_meta` (resume) is left
        alone; an empty file gets the header now."""
        if not self.run_id:
            return
        # Cheap check: if any line exists, header was already written
        # (or the operator deliberately omitted it on a legacy file).
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        meta = {
            "_meta": True,
            "run_id": self.run_id,
            "model": self.model,
            "code_version": self.code_version,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(meta) + "\n")
            f.flush()

    # ── reads ─────────────────────────────────────────────────────────

    def done_keys(self) -> set[str]:
        """Keys of episodes considered DONE for resume-skip purposes.

        Cells whose recorded `outcome == "error"` are EXCLUDED — an
        error means the provider/engine call faulted (429 storm,
        timeout, malformed tool-call response, …), not that the cell
        was actually played. Resuming should RETRY those, not skip.
        A real `win` / `loss` / `draw` cell stays in the done set.
        """
        if not self.path.exists():
            return set()
        keys: set[str] = set()
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001 — tolerate a torn last line
                continue
            if rec.get("_meta"):
                continue
            if rec.get("outcome") == "error":
                continue  # retry on next run
            key = rec.get("_key")
            if key is not None:
                keys.add(key)
        return keys

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if rec.get("_meta"):
                continue
            out.append(rec)
        return out

    def header(self) -> dict | None:
        """Return the `_meta` header dict if present, else None."""
        if not self.path.exists():
            return None
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if rec.get("_meta"):
                return rec
            # Header (if any) is always first; bail on first data row.
            return None
        return None

    # ── writes ────────────────────────────────────────────────────────

    def append(self, key: str, record: dict) -> None:
        row = dict(record)
        row["_key"] = key
        line = json.dumps(row)
        with self._lock:
            if key in self._seen_keys:
                raise DuplicateJournalKey(
                    f"key {key!r} appended twice within this process — "
                    f"likely a task-dup or resume-gate bug"
                )
            self._seen_keys.add(key)
            self._ensure_header()
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()
