#!/usr/bin/env python3
"""Production-eval orchestrator for the OpenRA-Bench paper baseline.

Launches `openra_bench.run_eval` for each (model, type) cell in the
campaign matrix, tracks progress under a single run-dir tree, and
auto-opens a PR to the paper repo when a cell completes.

Layout under `data/runs/v1.1-prod/`:

    manifest.json
    <model_slug>/
        scenarios/
            eval_stats.json
            journal__<model>.jsonl
            playback/
            status.json
        1v1/
            eval_stats.json
            ...

The orchestrator is intentionally a thin shell over `run_eval`:

  - It only EXECs / SUBPROCESSES `python3 -m openra_bench.run_eval`
    (no in-process import) so each cell gets a clean Python heap and
    a clean OpenRouter / OpenAI / Together client.
  - It NEVER writes to `eval_stats.json` itself — that file is the
    authoritative artefact of `run_eval`. The orchestrator only writes
    `manifest.json` (campaign-level) and `<cell>/status.json` (per-cell).
  - Resume safety: a re-launch reads `manifest.json` + each cell's
    `status.json` and inherits the "complete" state from disk. The
    `run_eval` side picks up its journal independently — we set the
    journal path deterministically per (out_dir, model).
  - Adaptive concurrency: the orchestrator computes an INITIAL value
    (default 20) and forwards `--concurrency N` plus `--strict-resume`
    to `run_eval`. The actual in-process adaptive backoff is handled
    by the parallel resume-hardening agent's `run_eval` changes; we
    only need to honour `--strict-resume` here.

Subcommands:

    launch       run one (model, type) cell
    status       print campaign progress
    campaign     run every (model, type) sequentially (scenarios-first)
    pr           open the auto-PR for a finished cell

NOTHING in this file touches `openra_bench/run_eval.py` — the parallel
agent owns it. We only call its CLI.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Campaign matrix
# ---------------------------------------------------------------------------

# (slug, provider, model_id). The slug is the FILESYSTEM-SAFE identity used
# in the run-dir tree; the model_id is what we forward to `--model`.
MODELS: tuple[tuple[str, str, str], ...] = (
    ("qwen3.5-9b",            "together",   "Qwen/Qwen3.5-9B"),
    ("gemma-4-31b-it",        "together",   "together_sso/google/gemma-4-31B-it-f5dbf8ad"),
    ("qwen3.6-35b-a3b",       "together",   "together_sso/Qwen/Qwen3.6-35B-A3B-FP8-46d45bad"),
    ("gpt-5.4-mini",          "openai",     "gpt-5.4-mini-2026-03-17"),
    ("gpt-5.4",               "openai",     "gpt-5.4-2026-03-05"),
    ("glm-4.6v",              "openrouter", "z-ai/glm-4.6v"),
)

TYPES: tuple[str, ...] = ("scenarios", "1v1")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROD_DIR = REPO_ROOT / "data" / "runs" / "v1.1-prod"
PAPER_REPO_URL = "https://github.com/KaiserWhoLearns/RedAlertBenchPaper"
PAPER_REPO_CLONE = Path("/tmp/RedAlertBenchPaper")


def _model_by_slug(slug: str) -> tuple[str, str, str]:
    for m in MODELS:
        if m[0] == slug:
            return m
    raise SystemExit(f"unknown model slug: {slug!r} (choices: "
                     f"{', '.join(m[0] for m in MODELS)})")


# ---------------------------------------------------------------------------
# Pack-family classification (kept in sync with tools/consolidate_results.py)
# ---------------------------------------------------------------------------

def pack_family(pack: str) -> str:
    """Map a pack name to its family folder label.

    Mirrors `tools/consolidate_results._pack_family` — kept here to avoid
    a cross-file import for what is a short pure function.
    """
    p = pack.lower()
    if p.startswith(("def-", "defense-")) or p.startswith("build-defensive-"):
        return "family3-defense"
    if p.startswith(("combat-", "action-", "harass-")):
        return "family1-combat-micro"
    if p.startswith(("econ-", "economy-")):
        return "family2-economy"
    if p.startswith(("scout-", "perception-", "navigation-")):
        return "family4-scout-perception"
    if p.startswith(("lh-", "longhorizon-")):
        return "family5-long-horizon"
    if p.startswith(("build-", "building-", "tech-", "power-")):
        return "family6-build-tech-power"
    if p.startswith(("proc-", "strict-", "maint-", "rob-")):
        return "family7-procedure-robustness"
    if p.startswith(("mfb-", "mcv-", "coord-", "coordination-")):
        return "family8-multi-front-coord"
    if p.startswith((
        "tp-", "tempo-", "strategy-", "adv-", "adversarial-",
        "artofwar-", "risk-", "reasoning-", "expansion-", "mid-",
    )):
        return "family9-tempo-strategy"
    return "family10-special-misc"


# ---------------------------------------------------------------------------
# Manifest / status I/O
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: dict) -> None:
    # Per-process temp suffix so concurrent launchers don't race on the
    # shared `manifest.json.tmp` filename (last writer's replace() would
    # FileNotFound on the others' tmps). PID + os.urandom yields a unique
    # path even under fork.
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}.{os.urandom(4).hex()}"
    tmp = path.with_suffix(path.suffix + suffix)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_manifest(prod_dir: Path) -> dict:
    mf = prod_dir / "manifest.json"
    existing = _read_json(mf) or {}
    if existing.get("schema") == "v1":
        return existing
    cells: dict[str, dict] = {}
    for slug, provider, model_id in MODELS:
        for t in TYPES:
            cells[f"{slug}:{t}"] = {
                "model_slug": slug,
                "provider": provider,
                "model_id": model_id,
                "type": t,
                "state": "not_started",
                "pr_url": None,
            }
    payload = {
        "schema": "v1",
        "campaign": "v1.1-prod",
        "started_at": _now_iso(),
        "cells": cells,
    }
    _atomic_write_json(mf, payload)
    return payload


def cell_dir(prod_dir: Path, slug: str, type_: str) -> Path:
    return prod_dir / slug / type_


def cell_status_path(prod_dir: Path, slug: str, type_: str) -> Path:
    return cell_dir(prod_dir, slug, type_) / "status.json"


def update_cell_state(prod_dir: Path, slug: str, type_: str,
                      **fields: Any) -> dict:
    """Merge `fields` into the cell's manifest record AND its status.json.

    Returns the updated manifest cell dict.
    """
    mf_path = prod_dir / "manifest.json"
    mf = load_manifest(prod_dir)
    key = f"{slug}:{type_}"
    cell = mf["cells"].setdefault(key, {
        "model_slug": slug, "type": type_, "state": "not_started",
    })
    cell.update(fields)
    cell["updated_at"] = _now_iso()
    _atomic_write_json(mf_path, mf)

    snap = dict(cell)
    snap["snapshot_at"] = _now_iso()
    _atomic_write_json(cell_status_path(prod_dir, slug, type_), snap)
    return cell


# ---------------------------------------------------------------------------
# run_eval invocation
# ---------------------------------------------------------------------------

def _check_run_eval_flag(flag: str) -> bool:
    """Return True if `python3 -m openra_bench.run_eval --help` mentions `flag`.

    Used to feature-detect `--strict-resume` so we don't crash on an
    older `run_eval` that hasn't yet been touched by the resume-hardening
    agent. We only PASS the flag when it's available.
    """
    try:
        cp = subprocess.run(
            [sys.executable, "-m", "openra_bench.run_eval", "--help"],
            check=False, capture_output=True, text=True, timeout=30,
            cwd=str(REPO_ROOT),
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return flag in (cp.stdout or "") or flag in (cp.stderr or "")


def build_run_eval_argv(*, slug: str, provider: str, model_id: str,
                        type_: str, out_dir: Path,
                        concurrency: int,
                        levels: str, seeds: str,
                        opponent: str = "scripted:stall",
                        extra: Iterable[str] = ()) -> list[str]:
    """Build the argv we hand to `python3 -m openra_bench.run_eval`.

    Mirrors `run_eval`'s argparse exactly — see openra_bench/run_eval.py
    around line 1186.
    """
    out_json = out_dir / "eval_stats.json"
    playback = out_dir / "playback"
    journal = out_dir / f"journal__{re.sub('[^A-Za-z0-9._-]', '_', model_id)}.jsonl"

    argv = [
        sys.executable, "-m", "openra_bench.run_eval",
        "--provider", provider,
        "--model", model_id,
        "--levels", levels,
        "--seeds", seeds,
        "--concurrency", str(concurrency),
        "--out", str(out_json),
        "--playback", str(playback),
        "--journal", str(journal),
        "--resume",
    ]
    if _check_run_eval_flag("--strict-resume"):
        argv.append("--strict-resume")
    if type_ == "1v1":
        argv += ["--mode", "1v1", "--opponent", opponent, "--side-swap"]
    argv += list(extra)
    return argv


def launch_cell(prod_dir: Path, slug: str, type_: str, *,
                concurrency: int = 20,
                levels: str = "easy,medium,hard",
                seeds: str = "1,2,3",
                opponent: str = "scripted:stall",
                extra: Iterable[str] = (),
                dry_run: bool = False) -> int:
    """Launch ONE cell. Blocks until run_eval exits.

    Returns the subprocess exit code (0 = success).
    """
    _slug, provider, model_id = _model_by_slug(slug)
    out_dir = cell_dir(prod_dir, slug, type_)
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = build_run_eval_argv(
        slug=slug, provider=provider, model_id=model_id, type_=type_,
        out_dir=out_dir, concurrency=concurrency,
        levels=levels, seeds=seeds, opponent=opponent, extra=extra,
    )

    update_cell_state(prod_dir, slug, type_,
                      state="in_progress",
                      started_at=_now_iso(),
                      cmd=" ".join(shlex.quote(a) for a in argv),
                      concurrency=concurrency,
                      levels=levels, seeds=seeds,
                      provider=provider, model_id=model_id)

    if dry_run:
        print("[dry-run]", " ".join(shlex.quote(a) for a in argv))
        update_cell_state(prod_dir, slug, type_, state="not_started",
                          dry_run=True)
        return 0

    log_path = out_dir / "run_eval.log"
    print(f"[launch] {slug} / {type_} -> {log_path}")
    with log_path.open("ab") as logf:
        logf.write(f"\n=== launched at {_now_iso()} ===\n".encode())
        logf.write(("# " + " ".join(shlex.quote(a) for a in argv) + "\n").encode())
        logf.flush()
        try:
            cp = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=str(REPO_ROOT))
            rc = cp.returncode
        except KeyboardInterrupt:
            update_cell_state(prod_dir, slug, type_, state="interrupted")
            raise

    state = "complete" if rc == 0 else "failed"
    update_cell_state(prod_dir, slug, type_, state=state,
                      finished_at=_now_iso(), exit_code=rc)
    return rc


# ---------------------------------------------------------------------------
# Status read-out
# ---------------------------------------------------------------------------

def _read_eval_stats(out_dir: Path) -> dict | None:
    return _read_json(out_dir / "eval_stats.json")


def _journal_lines(out_dir: Path) -> int:
    n = 0
    for p in out_dir.glob("journal__*.jsonl"):
        try:
            with p.open() as fh:
                for _ in fh:
                    n += 1
        except OSError:
            continue
    return n


def cell_status(prod_dir: Path, slug: str, type_: str) -> dict:
    """Synthesize a per-cell status snapshot. Reads from disk on every call so
    a crashed orchestrator restart gets accurate state."""
    out_dir = cell_dir(prod_dir, slug, type_)
    snap = _read_json(cell_status_path(prod_dir, slug, type_)) or {}
    state = snap.get("state", "not_started")
    stats = _read_eval_stats(out_dir)
    journal_n = _journal_lines(out_dir)

    overall = (stats or {}).get("overall") or {}
    episodes = (stats or {}).get("episodes") or []
    cells_done = len(episodes)

    last_outcome = episodes[-1]["outcome"] if episodes else None
    last_cell = episodes[-1]["cell"] if episodes else None

    # crude error rate: episodes flagged with `notes` containing "error"
    err = sum(1 for e in episodes
              if any("error" in str(n).lower() for n in (e.get("notes") or [])))
    error_rate = (err / cells_done) if cells_done else 0.0

    return {
        "model_slug": slug,
        "type": type_,
        "state": state,
        "cells_done": cells_done,
        "journal_lines": journal_n,
        "last_cell": last_cell,
        "last_outcome": last_outcome,
        "win_rate": overall.get("win_rate"),
        "composite_mean": overall.get("composite_mean"),
        "error_rate": round(error_rate, 4),
        "pr_url": snap.get("pr_url"),
        "started_at": snap.get("started_at"),
        "finished_at": snap.get("finished_at"),
    }


def render_status_table(prod_dir: Path) -> str:
    rows = []
    rows.append(("MODEL", "TYPE", "STATE", "DONE", "WIN", "COMP", "ERR", "LAST"))
    for slug, _provider, _model in MODELS:
        for t in TYPES:
            s = cell_status(prod_dir, slug, t)
            rows.append((
                slug, t, s["state"],
                str(s["cells_done"]),
                ("-" if s["win_rate"] is None else f"{s['win_rate']:.2f}"),
                ("-" if s["composite_mean"] is None else f"{s['composite_mean']:.2f}"),
                f"{s['error_rate']:.2%}",
                (s["last_cell"] or "-"),
            ))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(r))
        out.append(line)
        if i == 0:
            out.append("-" * len(line))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _safe(x: Any, n: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def summarise_run(stats: dict, *, slug: str, type_: str) -> dict:
    overall = stats.get("overall") or {}
    summary = stats.get("summary") or {}
    episodes = stats.get("episodes") or []

    # per-family roll-up. Provider-error cells are NOT model-failure data
    # — they are records of a 5xx / malformed-JSON / timeout where the
    # bench never got a model response to score. Counting them as
    # losses produces misleading 0% win-rate columns on families that
    # happened to fall during a provider outage. Track them separately
    # and compute win_rate over completed cells only (`n_eval = n -
    # errors`).
    fam_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "comp_sum": 0.0,
                 "leaves_block": 0, "errors": 0})
    for ep in episodes:
        pack = (ep.get("cell") or "").split(":", 1)[0]
        fam = pack_family(pack)
        f = fam_acc[fam]
        f["n"] += 1
        outcome = ep.get("outcome")
        if outcome == "error":
            f["errors"] += 1
            continue  # error cells excluded from win/composite stats
        f["wins"] += 1 if outcome == "win" else 0
        f["comp_sum"] += float(ep.get("composite") or 0.0)
        wl = ep.get("weakest_link")
        if wl and wl != "objective":
            f["leaves_block"] += 1
    fam_rows = []
    for fam, acc in sorted(fam_acc.items()):
        n = acc["n"]
        n_eval = n - acc["errors"]
        fam_rows.append({
            "family": fam,
            "n": n,
            "errors": acc["errors"],
            "n_eval": n_eval,
            "win_rate": (acc["wins"] / n_eval) if n_eval else None,
            "composite_mean": (acc["comp_sum"] / n_eval) if n_eval else None,
            "leaves_block_rate": (acc["leaves_block"] / n_eval) if n_eval else None,
        })

    # per-cell composite ranks (top 5 / bottom 5)
    cell_acc: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "comp_sum": 0.0, "wins": 0})
    for ep in episodes:
        c = ep.get("cell")
        if not c:
            continue
        cell_acc[c]["n"] += 1
        cell_acc[c]["comp_sum"] += float(ep.get("composite") or 0.0)
        cell_acc[c]["wins"] += 1 if ep.get("outcome") == "win" else 0
    cell_rows = []
    for c, acc in cell_acc.items():
        cell_rows.append({
            "cell": c,
            "n": acc["n"],
            "win_rate": (acc["wins"] / acc["n"]) if acc["n"] else 0.0,
            "composite_mean": (acc["comp_sum"] / acc["n"]) if acc["n"] else 0.0,
        })
    cell_rows.sort(key=lambda r: r["composite_mean"], reverse=True)
    best = cell_rows[:5]
    worst = list(reversed(cell_rows[-5:])) if len(cell_rows) >= 5 else []

    # spawn rotation (1v1 only)
    spawn_rows: list[dict] = []
    if type_ == "1v1":
        spawn_acc: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "wins": 0, "losses": 0, "draws": 0})
        for ep in episodes:
            sp = ep.get("spawn_point") or ep.get("seed")
            if sp is None:
                continue
            key = f"spawn-{sp}"
            spawn_acc[key]["n"] += 1
            o = ep.get("outcome")
            spawn_acc[key]["wins" if o == "win"
                           else "losses" if o == "loss" else "draws"] += 1
        for k, a in sorted(spawn_acc.items()):
            spawn_rows.append({
                "spawn": k, "n": a["n"],
                "win_rate": (a["wins"] / a["n"]) if a["n"] else 0.0,
                "wins": a["wins"], "losses": a["losses"], "draws": a["draws"],
            })

    n_eps = len(episodes)
    n_errors = sum(1 for e in episodes if e.get("outcome") == "error")
    n_wins = sum(1 for e in episodes if e.get("outcome") == "win")
    n_losses = sum(1 for e in episodes if e.get("outcome") == "loss")
    n_draws = sum(1 for e in episodes if e.get("outcome") == "draw")
    n_eval = n_eps - n_errors  # cells the model actually played
    leaves_block = sum(1 for e in episodes
                       if e.get("outcome") != "error"
                       and e.get("weakest_link")
                       and e.get("weakest_link") != "objective")

    return {
        "model_slug": slug,
        "type": type_,
        "episodes": n_eps,
        "errors": n_errors,
        "n_eval": n_eval,
        "wins": n_wins, "losses": n_losses, "draws": n_draws,
        "win_rate": (n_wins / n_eval) if n_eval else None,
        "composite_mean": overall.get("composite_mean"),
        "leaves_block_rate": (leaves_block / n_eval) if n_eval else None,
        "per_family": fam_rows,
        "best_cells": best,
        "worst_cells": worst,
        "spawn_rotation": spawn_rows,
        "model": stats.get("model"),
        "run_id": stats.get("run_id"),
    }


def render_summary_md(summary: dict) -> str:
    lines: list[str] = []
    L = lines.append
    errors = summary.get("errors", 0)
    n_eval = summary.get("n_eval", summary["episodes"] - errors)
    L(f"# {summary['model_slug']} / {summary['type']}")
    L("")
    L(f"- model: `{summary.get('model')}`")
    L(f"- run_id: `{summary.get('run_id')}`")
    L(f"- episodes: {summary['episodes']}  "
      f"(evaluated: {n_eval}, provider errors: {errors})")
    L(f"- wins / losses / draws: "
      f"{summary['wins']} / {summary['losses']} / {summary['draws']}")
    L(f"- win rate (over evaluated cells): {_safe(summary.get('win_rate'))}")
    L(f"- mean composite: {_safe(summary.get('composite_mean'))}")
    L(f"- leaves-final-blocking ratio: "
      f"{_safe(summary['leaves_block_rate'])}")
    if errors:
        L("")
        L(f"> **Note**: {errors} cells errored on the provider side "
          f"(5xx / malformed JSON / timeout). These are EXCLUDED from "
          f"win-rate and composite stats. Per-family `n` shows total cells; "
          f"`err` shows errored cells; rates are over `n - err`.")
    L("")
    L("## per family")
    L("")
    L("| family | n | err | win rate | composite | blocked |")
    L("| --- | ---:| ---:| ---:| ---:| ---:|")
    for r in summary["per_family"]:
        L(f"| {r['family']} | {r['n']} | {r.get('errors', 0)} | "
          f"{_safe(r['win_rate'])} | {_safe(r['composite_mean'])} | "
          f"{_safe(r['leaves_block_rate'])} |")
    if summary["best_cells"]:
        L("")
        L("## top 5 cells")
        L("")
        L("| cell | n | win | composite |")
        L("| --- | ---:| ---:| ---:|")
        for r in summary["best_cells"]:
            L(f"| {r['cell']} | {r['n']} | {_safe(r['win_rate'])} "
              f"| {_safe(r['composite_mean'])} |")
    if summary["worst_cells"]:
        L("")
        L("## bottom 5 cells")
        L("")
        L("| cell | n | win | composite |")
        L("| --- | ---:| ---:| ---:|")
        for r in summary["worst_cells"]:
            L(f"| {r['cell']} | {r['n']} | {_safe(r['win_rate'])} "
              f"| {_safe(r['composite_mean'])} |")
    if summary["spawn_rotation"]:
        L("")
        L("## spawn rotation (1v1)")
        L("")
        L("| spawn | n | wins | losses | draws | win rate |")
        L("| --- | ---:| ---:| ---:| ---:| ---:|")
        for r in summary["spawn_rotation"]:
            L(f"| {r['spawn']} | {r['n']} | {r['wins']} "
              f"| {r['losses']} | {r['draws']} | {_safe(r['win_rate'])} |")
    L("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper-repo PR
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path, check: bool = True,
         capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=check,
        capture_output=capture, text=capture,
    )


def ensure_paper_clone(clone_path: Path = PAPER_REPO_CLONE) -> Path | None:
    """Ensure the paper repo is cloned locally; return its path or None on
    failure (network / auth)."""
    try:
        if not clone_path.exists():
            cp = subprocess.run(
                ["git", "clone", PAPER_REPO_URL, str(clone_path)],
                check=False, capture_output=True, text=True, timeout=120,
            )
            if cp.returncode != 0:
                print(f"[pr] clone failed: {cp.stderr.strip()[:300]}",
                      file=sys.stderr)
                return None
        else:
            subprocess.run(["git", "-C", str(clone_path), "fetch", "origin"],
                           check=False, capture_output=True, timeout=60)
            subprocess.run(["git", "-C", str(clone_path),
                            "checkout", "main"],
                           check=False, capture_output=True, timeout=30)
            subprocess.run(["git", "-C", str(clone_path),
                            "reset", "--hard", "origin/main"],
                           check=False, capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[pr] clone/reset error: {e}", file=sys.stderr)
        return None
    return clone_path


def open_paper_pr(prod_dir: Path, slug: str, type_: str, *,
                  dry_run: bool = False) -> str | None:
    """Open a PR on the paper repo with the cell's eval_stats + summary md.

    Non-fatal: any failure prints + returns None instead of raising, so an
    orchestrator running the full campaign keeps going.
    """
    out_dir = cell_dir(prod_dir, slug, type_)
    stats = _read_eval_stats(out_dir)
    if stats is None:
        print(f"[pr] no eval_stats.json at {out_dir} — skipping",
              file=sys.stderr)
        return None

    summary = summarise_run(stats, slug=slug, type_=type_)
    md = render_summary_md(summary)

    if dry_run:
        print(md)
        return None

    repo = ensure_paper_clone()
    if repo is None:
        print(f"[pr] paper repo unavailable — saving summary locally",
              file=sys.stderr)
        (out_dir / "summary.md").write_text(md)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return None

    today = _dt.date.today().isoformat()
    branch = f"results-{slug}-{type_}-{today}"

    try:
        _git(["checkout", "-B", branch], cwd=repo)
        results_dir = repo / "results" / slug
        results_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "eval_stats.json", results_dir / f"{type_}.json")
        (results_dir / f"{type_}.md").write_text(md)
        (results_dir / f"{type_}.summary.json").write_text(
            json.dumps(summary, indent=2))

        _git(["add", str(results_dir.relative_to(repo))], cwd=repo)
        # Has any staged change?
        rc = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(repo), check=False,
        ).returncode
        if rc == 0:
            print(f"[pr] no diff for {slug}/{type_} — skipping commit",
                  file=sys.stderr)
            return None
        msg = f"results: {slug} on {type_}"
        _git(["commit", "-m", msg], cwd=repo)
        _git(["push", "-u", "origin", branch, "--force"], cwd=repo)

        # gh pr create
        title = f"results: {slug} on {type_}"
        body = md
        cp = subprocess.run(
            ["gh", "pr", "create", "--title", title, "--body", body,
             "--head", branch, "--base", "main"],
            cwd=str(repo), check=False, capture_output=True, text=True,
            timeout=120,
        )
        if cp.returncode != 0:
            # PR may already exist — view it instead.
            view = subprocess.run(
                ["gh", "pr", "view", branch, "--json", "url",
                 "-q", ".url"],
                cwd=str(repo), check=False, capture_output=True, text=True,
            )
            url = (view.stdout or "").strip() or None
            if url is None:
                print(f"[pr] gh pr create failed: "
                      f"{cp.stderr.strip()[:300]}", file=sys.stderr)
                return None
        else:
            url = (cp.stdout or "").strip().splitlines()[-1] if cp.stdout else None

        update_cell_state(prod_dir, slug, type_, pr_url=url)
        return url
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[pr] error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--prod-dir", default=str(DEFAULT_PROD_DIR),
                    help="campaign run dir (default: %(default)s)")


def cmd_launch(args: argparse.Namespace) -> int:
    prod = Path(args.prod_dir)
    load_manifest(prod)
    rc = launch_cell(
        prod, args.model, args.type,
        concurrency=args.concurrency,
        levels=args.levels, seeds=args.seeds,
        opponent=args.opponent,
        dry_run=args.dry_run,
    )
    if rc == 0 and args.auto_pr and not args.dry_run:
        url = open_paper_pr(prod, args.model, args.type)
        if url:
            print(f"[pr] {url}")
    return rc


def cmd_status(args: argparse.Namespace) -> int:
    prod = Path(args.prod_dir)
    load_manifest(prod)
    print(render_status_table(prod))
    return 0


def cmd_campaign(args: argparse.Namespace) -> int:
    prod = Path(args.prod_dir)
    load_manifest(prod)

    # scenarios first for every model, THEN 1v1 (paper-baseline order).
    ordered: list[tuple[str, str]] = []
    for t in ("scenarios", "1v1"):
        for slug, _p, _m in MODELS:
            ordered.append((slug, t))

    for slug, t in ordered:
        # gating: never launch a model's 1v1 cell until its scenarios cell
        # has been marked complete.
        if t == "1v1":
            sc = cell_status(prod, slug, "scenarios")
            if sc["state"] != "complete":
                print(f"[campaign] skip {slug}/1v1 — scenarios not complete "
                      f"(state={sc['state']})")
                continue
        st = cell_status(prod, slug, t)
        if st["state"] == "complete":
            print(f"[campaign] skip {slug}/{t} — already complete")
            continue
        rc = launch_cell(
            prod, slug, t, concurrency=args.concurrency,
            levels=args.levels, seeds=args.seeds,
            opponent=args.opponent, dry_run=args.dry_run,
        )
        if rc == 0 and args.auto_pr and not args.dry_run:
            url = open_paper_pr(prod, slug, t)
            if url:
                print(f"[pr] {url}")
    print(render_status_table(prod))
    return 0


def cmd_pr(args: argparse.Namespace) -> int:
    prod = Path(args.prod_dir)
    load_manifest(prod)
    url = open_paper_pr(prod, args.model, args.type, dry_run=args.dry_run)
    if url:
        print(url)
        return 0
    return 1 if not args.dry_run else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_production_eval",
        description="Production-eval orchestrator for the OpenRA-Bench "
                    "paper baseline (5 models x 2 modes; auto-PRs to "
                    "RedAlertBenchPaper).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_launch = sub.add_parser("launch", help="launch ONE (model, type) cell")
    _add_common(p_launch)
    p_launch.add_argument("--model", required=True,
                          choices=[m[0] for m in MODELS])
    p_launch.add_argument("--type", required=True, choices=list(TYPES))
    p_launch.add_argument("--concurrency", type=int, default=20)
    p_launch.add_argument("--levels", default="easy,medium,hard")
    p_launch.add_argument("--seeds", default="1,2,3")
    p_launch.add_argument("--opponent", default="scripted:stall",
                          help="1v1 opponent spec (ignored for scenarios)")
    p_launch.add_argument("--auto-pr", action="store_true",
                          help="open a paper-repo PR on successful exit")
    p_launch.add_argument("--dry-run", action="store_true")
    p_launch.set_defaults(fn=cmd_launch)

    p_status = sub.add_parser("status", help="print campaign progress")
    _add_common(p_status)
    p_status.set_defaults(fn=cmd_status)

    p_camp = sub.add_parser("campaign", help="run every cell sequentially")
    _add_common(p_camp)
    p_camp.add_argument("--concurrency", type=int, default=20)
    p_camp.add_argument("--levels", default="easy,medium,hard")
    p_camp.add_argument("--seeds", default="1,2,3")
    p_camp.add_argument("--opponent", default="scripted:stall")
    p_camp.add_argument("--auto-pr", action="store_true")
    p_camp.add_argument("--dry-run", action="store_true")
    p_camp.set_defaults(fn=cmd_campaign)

    p_pr = sub.add_parser("pr", help="open the paper-repo PR for a cell")
    _add_common(p_pr)
    p_pr.add_argument("--model", required=True,
                      choices=[m[0] for m in MODELS])
    p_pr.add_argument("--type", required=True, choices=list(TYPES))
    p_pr.add_argument("--dry-run", action="store_true",
                      help="print the summary markdown, don't push")
    p_pr.set_defaults(fn=cmd_pr)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
