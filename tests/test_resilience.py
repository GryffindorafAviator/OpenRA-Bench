"""Resilience layer: retry/backoff, throttle, cost cap, journal/resume,
bounded history, dry-run/smoke. All deterministic — fake clocks/sleeps,
no network, scripted agent for the evaluate-integration paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.resilience import (
    BudgetExceeded,
    CostMeter,
    FatalProviderError,
    RateLimiter,
    RetryPolicy,
    RunJournal,
    episode_key,
    retry_call,
)


# ── retry / backoff ────────────────────────────────────────────────────────


def test_retry_policy_delay_exponential_capped_and_retry_after():
    # Disable jitter to assert the deterministic exponential backoff curve.
    p = RetryPolicy(base=1.0, cap=30.0, max_attempts=6, jitter=0.0)
    assert [p.delay(a) for a in (1, 2, 3, 4, 10)] == [1.0, 2.0, 4.0, 8.0, 30.0]
    # server Retry-After wins when larger; still capped
    assert p.delay(1, retry_after=5.0) == 5.0
    assert p.delay(1, retry_after=999) == 30.0
    assert p.is_transient_status(429) and not p.is_transient_status(400)


def test_retry_policy_jitter_spreads_concurrent_retries():
    """Regression: under high concurrency N workers hitting 429 at the
    same instant must NOT all back off by an identical interval (the
    pre-fix bug — `jitter` field existed but was `* 0`'d in `delay()`,
    producing a thundering-herd pattern on every retry attempt). With
    jitter active, the delay for a fixed attempt is sampled uniformly
    from a ±10% window around the deterministic backoff, so a fleet of
    20 workers fans out instead of stampeding."""
    import random as _r
    rng = _r.Random(42)
    p = RetryPolicy(base=4.0, cap=30.0, jitter=0.1)
    # Sample 200 retries at attempt=2 (deterministic backoff=8.0)
    samples = [p.delay(2, _rng=rng) for _ in range(200)]
    # All in the ±10% window
    assert all(7.19 < s < 8.81 for s in samples), \
        f"samples outside ±10%: min={min(samples)} max={max(samples)}"
    # Spread is non-trivial (at least 50 distinct values out of 200)
    assert len(set(samples)) > 50, \
        f"jitter not active — only {len(set(samples))} unique delays"


def test_retry_call_succeeds_after_transient_then_stops_on_fatal():
    slept = []
    n = {"i": 0}

    def flaky():
        n["i"] += 1
        if n["i"] < 3:
            e = RuntimeError("503"); e.transient = True
            raise e
        return "ok"

    assert retry_call(flaky, RetryPolicy(max_attempts=5),
                       sleep=slept.append) == "ok"
    assert len(slept) == 2  # two backoffs before the 3rd attempt

    def fatal():
        e = FatalProviderError("400"); e.transient = False
        raise e

    with pytest.raises(FatalProviderError):
        retry_call(fatal, RetryPolicy(max_attempts=5), sleep=slept.append)

    def always():
        e = RuntimeError("503"); e.transient = True
        raise e

    with pytest.raises(RuntimeError):
        retry_call(always, RetryPolicy(max_attempts=3), sleep=lambda *_: None)


# ── rate limiter ───────────────────────────────────────────────────────────


def test_rate_limiter_enforces_min_interval():
    clk = {"t": 0.0}
    slept = []
    rl = RateLimiter(qps=2.0)  # 0.5s spacing

    def now():
        return clk["t"]

    def slp(s):
        slept.append(s)
        clk["t"] += s

    assert rl.acquire(now=now, sleep=slp) == 0.0       # first is free
    w = rl.acquire(now=now, sleep=slp)                  # immediate 2nd waits
    assert w == pytest.approx(0.5)
    assert RateLimiter(0.0).acquire() == 0.0            # disabled


# ── cost meter ─────────────────────────────────────────────────────────────


def test_cost_meter_accumulates_and_caps():
    m = CostMeter(price_in_per_m=1.0, price_out_per_m=2.0, max_usd=0.01)
    m.add(1000, 1000)  # 0.001 + 0.002 = 0.003
    m.check()  # under cap
    assert m.usd == pytest.approx(0.003)
    m.add(2_000_000, 2_000_000)  # blows the cap
    with pytest.raises(BudgetExceeded):
        m.check()
    assert m.snapshot()["calls"] == 2


# ── journal / resume ───────────────────────────────────────────────────────


def test_journal_roundtrip_and_torn_line(tmp_path):
    j = RunJournal(tmp_path / "j.jsonl")
    assert j.done_keys() == set()
    k = episode_key("p", "easy", "public", 1)
    j.append(k, {"cell": "p:easy", "composite": 0.4})
    j.append(episode_key("p", "hard", "public", 2), {"cell": "p:hard"})
    assert k in j.done_keys() and len(j.done_keys()) == 2
    with open(tmp_path / "j.jsonl", "a") as f:
        f.write('{"_key": "torn"')  # no newline / invalid
    assert len(j.records()) == 2  # torn tail tolerated


def test_journal_error_row_allows_retry_append(tmp_path):
    """Regression: an errored row in the journal must NOT prevent a retry
    from being appended. Pre-fix the in-memory `_seen_keys` was seeded
    from EVERY prior row including errors, so when the resume gate
    correctly retried an errored cell the second append raised
    `DuplicateJournalKey` and crashed the launcher. Mirror `done_keys()`:
    error rows are excluded from the dedupe set so retries succeed."""
    jp = tmp_path / "j.jsonl"
    j = RunJournal(jp)
    k = episode_key("scout-jeep", "easy", "public", 1, "vision", repeat=1)
    # First attempt: provider 500 ⇒ recorded as error.
    j.append(k, {"cell": "scout-jeep:easy", "outcome": "error",
                 "notes": ["FatalProviderError: 500"]})
    # Re-open the journal as a fresh process would (resume-from-crash).
    j2 = RunJournal(jp)
    # done_keys excludes errors — confirms retry is intended.
    assert k not in j2.done_keys()
    # The retry's append MUST succeed, not raise DuplicateJournalKey.
    j2.append(k, {"cell": "scout-jeep:easy", "outcome": "win",
                  "composite": 0.85})
    # Both rows persist; done_keys returns the key once (the success).
    assert k in j2.done_keys()
    # Records returns both rows (caller dedups by `_key` keeping latest).
    rows = [r for r in j2.records() if r.get("_key") == k]
    assert len(rows) == 2
    assert rows[0]["outcome"] == "error"
    assert rows[1]["outcome"] == "win"


# ── bounded chat history ───────────────────────────────────────────────────


def test_model_agent_window_keeps_system_and_last_turns():
    from openra_bench.agent import ModelAgent

    h = [{"role": "system", "content": "S"}]
    for t in range(5):
        h.append({"role": "user", "content": f"u{t}"})
        h.append({"role": "assistant", "content": f"a{t}",
                  "tool_calls": [{"id": f"c{t}"}]})
        h.append({"role": "tool", "tool_call_id": f"c{t}", "content": "ok"})

    w = ModelAgent._window(h, 2)
    assert w[0]["role"] == "system"
    users = [m for m in w if m["role"] == "user"]
    assert [m["content"] for m in users] == ["u3", "u4"]  # last 2 groups
    # pairing intact: first non-system is a user (no dangling tool reply)
    assert w[1]["role"] == "user"
    # no trimming when within budget; passthrough when disabled
    assert ModelAgent._window(h, 99) is h
    assert ModelAgent._window(h, 0) is h


# ── evaluate: dry-run / smoke / journal+resume (scripted, no API) ───────────


def test_evaluate_dry_run_lists_without_running():
    from openra_bench.run_eval import evaluate

    PACK = Path("openra_bench/scenarios/packs/perception-frontier-reading.yaml")
    out = evaluate([PACK], ["easy", "medium"], [1], dry_run=True)
    assert out["dry_run"] and out["tasks"] == 2
    assert "episodes" not in out  # nothing executed


def test_evaluate_journal_resume_is_lossless(tmp_path):
    pytest.importorskip("openra_train")
    from openra_bench.run_eval import evaluate

    PACK = Path("openra_bench/scenarios/packs/perception-frontier-reading.yaml")
    jp = tmp_path / "j.jsonl"
    a = evaluate([PACK], ["easy"], [1, 2], journal_path=jp)
    assert a["overall"]["n"] == 2 and a["resumed"] == 0
    # Two data rows; a `_meta` header line may also be present (v11
    # production-eval hardening). Count by filtering data rows.
    rows = [
        ln for ln in jp.read_text().splitlines()
        if ln.strip() and not json.loads(ln).get("_meta")
    ]
    assert len(rows) == 2

    # resume: both episodes already journaled → 0 new, same aggregate
    b = evaluate([PACK], ["easy"], [1, 2], journal_path=jp, resume=True,
                 ignore_run_id=True)
    assert b["resumed"] == 2 and b["overall"]["n"] == 2
    rows2 = [
        ln for ln in jp.read_text().splitlines()
        if ln.strip() and not json.loads(ln).get("_meta")
    ]
    assert len(rows2) == 2  # nothing re-appended
    assert "cost" in b and "truncated" in b


# ── wire: tool_call arguments must be a JSON string (OpenRouter 400) ────────


def test_wire_messages_serializes_tool_call_arguments():
    from openra_bench.providers import OpenAICompatibleProvider as P

    hist = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "", "reasoning": "drop me",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "move_units",
                                      "arguments": {"unit_ids": [1],
                                                    "target_x": 5}}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "ok"},
    ]
    wire = P._wire_messages(hist)
    args = wire[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)            # JSON string, not dict
    assert json.loads(args) == {"unit_ids": [1], "target_x": 5}
    assert "reasoning" not in wire[1]       # playback-only key stripped
    # pure: original history untouched (still a dict for playback)
    assert isinstance(
        hist[1]["tool_calls"][0]["function"]["arguments"], dict
    )
    # already-string args are left alone
    hist[1]["tool_calls"][0]["function"]["arguments"] = '{"a":1}'
    assert P._wire_messages(hist)[1]["tool_calls"][0]["function"][
        "arguments"
    ] == '{"a":1}'


def test_evaluate_continues_past_a_failing_episode(tmp_path):
    pytest.importorskip("openra_train")
    from openra_bench.run_eval import evaluate

    PACK = Path("openra_bench/scenarios/packs/perception-frontier-reading.yaml")

    def boom_factory(_compiled):
        def agent_fn(_rs, _Command):
            raise RuntimeError("simulated fatal provider 400")
        return agent_fn

    out = evaluate([PACK], ["easy"], [1, 2], agent_factory=boom_factory,
                   journal_path=tmp_path / "j.jsonl")
    eps = out["episodes"]
    assert len(eps) == 2
    assert all(e["outcome"] == "error" for e in eps)   # recorded, not raised
    assert "overall" in out                            # report still produced
    # journal captured them so --resume won't re-run the errored ones
    # (count data rows only; v11 hardening prepends a `_meta` header)
    rows = [
        ln for ln in (tmp_path / "j.jsonl").read_text().splitlines()
        if ln.strip() and not json.loads(ln).get("_meta")
    ]
    assert len(rows) == 2
