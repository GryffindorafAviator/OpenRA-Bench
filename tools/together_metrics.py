"""Together AI provider observability probe.

Surfaces whatever Together exposes programmatically so an operator can
answer "is the LLM provider slow right now, or is it our code?" during
a live eval sweep.

Read-only. Does NOT restart endpoints, does NOT touch running sweeps.
End-to-end runtime target: <30s. Re-runnable.

Usage:
    set -a && source .env && set +a
    python3 tools/together_metrics.py
    python3 tools/together_metrics.py --json  # machine-readable

What it queries (sources of truth):
  1. SDK: client.endpoints.list(mine=True) + .retrieve(id) — state,
     autoscaling (min/max/current/ready replicas), hardware,
     inactive_timeout, availability_zone.
  2. SDK with_raw_response: per-call HTTP headers — server-side
     timing markers (x-api-received, x-api-call-start,
     x-api-call-end), request-id, cf-ray, inference engine version.
  3. Probe completion (max_tokens=4) per model: client-observed
     wall-clock latency + server-reported processing window.

What Together DOES expose on serverless models (rich, surprised):
  - x-ratelimit-limit / x-ratelimit-remaining (requests per minute)
  - x-ratelimit-limit-tokens / x-ratelimit-remaining-tokens
  - x-ratelimit-limit-dynamic / -remaining-dynamic (per-key
    concurrent in-flight lane — the "burst" budget)
  - x-ratelimit-reset (seconds until window resets)
  - retry-after on 2xx (advisory backoff hint)

What Together does NOT expose (confirmed by probing 2026-05-25):
  - No /v1/usage, /v1/billing, /v1/quotas, /v1/metrics, /v1/me,
    /v1/account endpoints (all 404).
  - No x-ratelimit-* headers on DEDICATED endpoint responses
    (capacity is bounded by replicas, not per-key quota).
  - No per-endpoint queue depth or requests-in-flight counter.
  - No live tokens/sec counter (only end-to-end wall time).
  - No per-hour spend tally via API.

So this probe is "best you can do today" — endpoint state +
autoscaling replicas + per-call latency breakdown. If a sweep
suddenly slows, an operator runs this once: a STOPPED dedicated
endpoint, replicas != ready_replicas, or x-api-call latency >>
client latency indicates "Together side", everything healthy here
indicates "our side".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any

import together


# v1.1 paper-baseline production models. Names match the spec the
# operator gave us; we resolve them at runtime against
# endpoints.list(mine=True) so a rename doesn't silently break this.
PRODUCTION_MODELS = [
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-31B-it-f5dbf8ad",
    "Qwen/Qwen3.6-35B-A3B-FP8-46d45bad",
]


@dataclass
class EndpointSnap:
    """One endpoint's read-only metadata + scaling snapshot."""

    id: str
    model: str
    name: str
    state: str
    type: str
    hardware: str | None
    availability_zone: str | None
    inactive_timeout_min: int | None
    min_replicas: int | None
    max_replicas: int | None
    current_replicas: int | None
    ready_replicas: int | None

    @property
    def replicas_ok(self) -> bool:
        if self.current_replicas is None or self.ready_replicas is None:
            return True  # serverless: nothing to assert
        return self.current_replicas == self.ready_replicas


@dataclass
class CallProbe:
    """One probe completion's timing + provider trace."""

    model: str
    status: int
    client_latency_ms: float
    server_latency_ms: float | None  # x-api-call-end - x-api-call-start
    queue_ms: float | None  # x-api-call-start - x-api-received
    request_id: str | None
    cf_ray: str | None
    inference_version: str | None
    ratelimit_headers: dict[str, str]  # any x-ratelimit-* / retry-after
    error: str | None = None


def resolve_endpoint(client: together.Together, needle: str) -> Any | None:
    """Find an endpoint by id substring OR by exact .model / .name match."""
    eps = client.endpoints.list(mine=True)
    matches = []
    for e in eps.data:
        hay = " ".join([e.id or "", e.model or "", e.name or ""])
        if needle in hay:
            matches.append(e)
    if not matches:
        return None
    # Prefer STARTED, then dedicated, then newest.
    matches.sort(
        key=lambda e: (
            0 if e.state == "STARTED" else 1,
            0 if e.type == "dedicated" else 1,
        )
    )
    return matches[0]


def snap_endpoint(client: together.Together, needle: str) -> EndpointSnap | None:
    e = resolve_endpoint(client, needle)
    if e is None:
        return None
    # .retrieve() carries autoscaling; .list() does NOT.
    try:
        full = client.endpoints.retrieve(e.id)
    except Exception:
        full = e
    auto = getattr(full, "autoscaling", None)
    return EndpointSnap(
        id=full.id,
        model=full.model or "",
        name=full.name or "",
        state=full.state or "",
        type=full.type or "",
        hardware=getattr(full, "hardware", None),
        availability_zone=getattr(full, "availability_zone", None),
        inactive_timeout_min=getattr(full, "inactive_timeout", None),
        min_replicas=getattr(auto, "min_replicas", None) if auto else None,
        max_replicas=getattr(auto, "max_replicas", None) if auto else None,
        current_replicas=getattr(auto, "current_replicas", None) if auto else None,
        ready_replicas=getattr(auto, "ready_replicas", None) if auto else None,
    )


def _parse_iso_ms(s: str | None) -> float | None:
    if not s:
        return None
    # Headers come back as "2026-05-25T16:23:50.354Z"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.timestamp() * 1000.0
    except Exception:
        return None


def probe_call(client: together.Together, model: str) -> CallProbe:
    t0 = time.perf_counter()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
        )
    except Exception as exc:
        return CallProbe(
            model=model,
            status=-1,
            client_latency_ms=(time.perf_counter() - t0) * 1000.0,
            server_latency_ms=None,
            queue_ms=None,
            request_id=None,
            cf_ray=None,
            inference_version=None,
            ratelimit_headers={},
            error=f"{type(exc).__name__}: {exc}",
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    resp = raw.http_response
    h = resp.headers
    received = _parse_iso_ms(h.get("x-api-received"))
    start = _parse_iso_ms(h.get("x-api-call-start"))
    end = _parse_iso_ms(h.get("x-api-call-end"))
    server_lat = (end - start) if (start is not None and end is not None) else None
    queue = (start - received) if (start is not None and received is not None) else None
    rl = {k: v for k, v in h.items() if k.lower().startswith(("x-ratelimit", "x-quota", "ratelimit-")) or k.lower() == "retry-after"}
    return CallProbe(
        model=model,
        status=resp.status_code,
        client_latency_ms=wall_ms,
        server_latency_ms=server_lat,
        queue_ms=queue,
        request_id=h.get("x-request-id"),
        cf_ray=h.get("cf-ray"),
        inference_version=h.get("x-inference-version"),
        ratelimit_headers=rl,
    )


def fmt_snap(s: EndpointSnap | None, needle: str) -> str:
    if s is None:
        return f"  [NOT FOUND] {needle}"
    rep = (
        f"{s.current_replicas}/{s.ready_replicas} (min={s.min_replicas} max={s.max_replicas})"
        if s.current_replicas is not None
        else "n/a (serverless)"
    )
    flag = "" if s.replicas_ok else "  !!REPLICAS NOT READY!!"
    return (
        f"  {s.name}\n"
        f"    id={s.id}\n"
        f"    state={s.state}  type={s.type}  hw={s.hardware}  az={s.availability_zone}\n"
        f"    inactive_timeout_min={s.inactive_timeout_min}  replicas={rep}{flag}"
    )


def fmt_probe(p: CallProbe) -> str:
    if p.error:
        return (
            f"  {p.model}\n"
            f"    ERROR after {p.client_latency_ms:.0f}ms: {p.error}"
        )
    sv = f"{p.server_latency_ms:.0f}ms" if p.server_latency_ms is not None else "n/a"
    q = f"{p.queue_ms:.0f}ms" if p.queue_ms is not None else "n/a"
    rl = (
        " ".join(f"{k}={v}" for k, v in p.ratelimit_headers.items())
        if p.ratelimit_headers
        else "(none returned)"
    )
    return (
        f"  {p.model}  HTTP {p.status}\n"
        f"    wall={p.client_latency_ms:.0f}ms  server={sv}  queue={q}\n"
        f"    request_id={p.request_id}  cf_ray={p.cf_ray}  engine={p.inference_version}\n"
        f"    ratelimit_headers={rl}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="endpoint snapshot only, no chat completion (read-only of read-only)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=PRODUCTION_MODELS,
        help="override the model list to probe",
    )
    args = parser.parse_args()

    if not os.environ.get("TOGETHER_API_KEY"):
        print("ERROR: TOGETHER_API_KEY not in env. Run: set -a; source .env; set +a", file=sys.stderr)
        return 2

    client = together.Together()
    t_total = time.perf_counter()

    if not args.json:
        print("=== Together AI provider snapshot ===")
        print(f"probed_at_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        print(f"sdk_version={together.__version__}")
        print()

    snaps: list[tuple[str, EndpointSnap | None]] = []
    for needle in args.models:
        s = snap_endpoint(client, needle)
        snaps.append((needle, s))

    if not args.json:
        print("--- endpoints ---")
        for needle, s in snaps:
            print(fmt_snap(s, needle))
        print()

    probes: list[CallProbe] = []
    if not args.skip_probe:
        for needle, s in snaps:
            # Dedicated endpoints route by their unique `.name` (the
            # `together_sso/...` handle). Serverless models route by
            # `.model`. Operator-supplied needle is the fallback.
            if s and s.type == "dedicated":
                model = s.name or s.model or needle
            elif s:
                model = s.model or needle
            else:
                model = needle
            probes.append(probe_call(client, model))

        if not args.json:
            print("--- probe completions (max_tokens=4) ---")
            for p in probes:
                print(fmt_probe(p))
            print()

    total_ms = (time.perf_counter() - t_total) * 1000.0
    if args.json:
        out = {
            "probed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sdk_version": together.__version__,
            "total_wall_ms": round(total_ms, 1),
            "endpoints": [
                {"needle": n, **(asdict(s) if s else {"missing": True})}
                for n, s in snaps
            ],
            "probes": [asdict(p) for p in probes],
            "available_on_serverless_only": [
                "x-ratelimit-limit / x-ratelimit-remaining (rpm)",
                "x-ratelimit-limit-tokens / x-ratelimit-remaining-tokens",
                "x-ratelimit-limit-dynamic / x-ratelimit-remaining-dynamic (in-flight lane)",
                "x-ratelimit-reset (seconds until window resets)",
            ],
            "not_available": [
                "No /v1/usage, /v1/billing, /v1/quotas, /v1/metrics endpoints (all 404).",
                "No x-ratelimit-* on dedicated endpoints (capacity is bounded by replicas).",
                "No queue depth / requests-in-flight / tokens-per-second counter.",
                "No per-hour spend via API; check the web console.",
            ],
        }
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"--- diagnostic ---")
        print(f"total_wall_ms={total_ms:.0f}")
        print()
        print("Exposed (serverless only):")
        print("  - x-ratelimit-{limit,remaining,reset} (rpm window)")
        print("  - x-ratelimit-{limit,remaining}-tokens")
        print("  - x-ratelimit-{limit,remaining}-dynamic (in-flight lane)")
        print()
        print("NOT exposed by Together (confirmed by 2026-05-25 probing):")
        print("  - No /v1/usage, /v1/billing, /v1/quotas, /v1/metrics (404).")
        print("  - No x-ratelimit-* on dedicated endpoints (capacity = replicas).")
        print("  - No queue depth / in-flight / tokens-per-second counter.")
        print("  - No per-hour spend via API; use the web console.")
        print()
        print("Operator triage:")
        print("  * endpoint STOPPED  -> dedicated endpoint cold; first hit pays cold-start.")
        print("  * current_replicas < ready_replicas  -> scaling up; transient slowdown.")
        print("  * server_latency_ms >> typical baseline  -> Together-side slowdown.")
        print("  * client wall - server_latency >> 200ms  -> our network or our code.")
        print("  * client error (-1)  -> see retry log in stderr of the sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
