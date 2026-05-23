"""OpenRA-Bench: Agent Leaderboard for OpenRA-RL.

A Gradio app that displays agent rankings, supports filtering by type
and opponent difficulty, and lets users run evaluations in-browser.

Run locally:
    python app.py

Deploy on HuggingFace Spaces:
    Push app.py, requirements.txt, data/, and README.md to your HF Space.
"""

import csv
import html
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import gradio_client.utils as _gc_utils
import pandas as pd

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
def _patched_json_schema_to_python_type(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)
_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

from evaluate_runner import DIFFICULTY_MULTIPLIER, DEFAULT_SERVER, compute_composite_score, compute_game_metrics

logger = logging.getLogger(__name__)

# ── Data Loading ──────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "results.csv"

AGENT_TYPE_COLORS = {
    "Scripted": "#ffcd75",  # Gold
    "LLM": "#7497db",       # Blue
    "RL": "#75809c",        # Gray-blue
}

DISPLAY_COLUMNS = [
    "Rank",
    "Agent",
    "Type",
    "Status",
    "Opponent",
    "Games",
    "Win Rate (%)",
    "Score",
    "K/D Ratio",
    "Avg Kills",
    "Avg Deaths",
    "Avg Economy",
    "Avg Game Length",
    "Date",
    "Replay",
]


def _safe_agent_link(name: str, url) -> str:
    """Render agent name, optionally as a hyperlink. HTML-escaped to prevent XSS."""
    safe_name = html.escape(str(name))
    if pd.notna(url) and str(url).strip():
        url_str = str(url).strip()
        # Only allow http/https URLs — block javascript:, data:, etc.
        if url_str.startswith(("http://", "https://")):
            safe_url = html.escape(url_str, quote=True)
            return f'<a href="{safe_url}" target="_blank" rel="noopener">{safe_name}</a>'
    return safe_name


def _verified_badge(verified) -> str:
    """Render a Verified/Unverified HTML badge."""
    if isinstance(verified, str):
        verified = verified.lower() in ("true", "1", "yes")
    if verified:
        return (
            '<span style="background:#4caf50;color:#fff;'
            'padding:2px 8px;border-radius:4px;font-size:0.85em">'
            'Verified</span>'
        )
    return (
        '<span style="background:#ff9800;color:#fff;'
        'padding:2px 8px;border-radius:4px;font-size:0.85em">'
        'Unverified</span>'
    )


def _safe_replay_link(url) -> str:
    """Render replay download link. Filename is sanitized to prevent XSS."""
    if pd.notna(url) and str(url).strip():
        # Sanitize: only allow alphanumeric, dash, underscore, dot
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "", str(url).strip())
        if safe_name:
            escaped = html.escape(safe_name, quote=True)
            return f'<a href="/replays/{escaped}" download title="Download replay">&#11015;</a>'
    return ""


def load_data() -> pd.DataFrame:
    """Load leaderboard data from CSV."""
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Build agent name with optional hyperlink (XSS-safe)
    if "agent_url" in df.columns:
        df["Agent"] = df.apply(
            lambda r: _safe_agent_link(r.get("agent_name", ""), r.get("agent_url", "")),
            axis=1,
        )
    else:
        df["Agent"] = df["agent_name"].apply(lambda n: html.escape(str(n)))

    # Build replay download link (XSS-safe)
    if "replay_url" in df.columns:
        df["Replay"] = df["replay_url"].apply(_safe_replay_link)
    else:
        df["Replay"] = ""

    # Verified/Unverified badge
    if "verified" in df.columns:
        df["Status"] = df["verified"].apply(_verified_badge)
    else:
        df["Status"] = _verified_badge(True)  # Legacy data = verified

    # Rename for display
    df = df.rename(columns={
        "agent_type": "Type",
        "opponent": "Opponent",
        "games": "Games",
        "win_rate": "Win Rate (%)",
        "score": "Score",
        "kd_ratio": "K/D Ratio",
        "avg_kills": "Avg Kills",
        "avg_deaths": "Avg Deaths",
        "avg_economy": "Avg Economy",
        "avg_game_length": "Avg Game Length",
        "timestamp": "Date",
    })

    return df[DISPLAY_COLUMNS]


def add_type_badges(df: pd.DataFrame) -> pd.DataFrame:
    """Add color-coded HTML badges to the Type column."""
    def badge(agent_type: str) -> str:
        color = AGENT_TYPE_COLORS.get(agent_type, "#ccc")
        text_color = "#fff" if agent_type != "Scripted" else "#333"
        return (
            f'<span style="background:{color};color:{text_color};'
            f'padding:2px 8px;border-radius:4px;font-size:0.85em">'
            f"{agent_type}</span>"
        )

    df = df.copy()
    df["Type"] = df["Type"].apply(badge)
    return df


def load_capability_leaderboard() -> pd.DataFrame:
    """Ranked capability leaderboard from the run_eval JSONL store
    (composite + Perception/Reasoning/Action + dominant weakest link)."""
    try:
        from openra_bench.leaderboard import build_table

        rows = build_table()
    except Exception:  # noqa: BLE001 — never break the UI on a bad store
        rows = []
    cols = [
        "rank", "model", "episodes", "win_rate", "composite",
        "objective", "adversarial_rating", "perception", "reasoning",
        "action", "weakest_link", "reward_vector",
        "held_out_composite", "generalization_gap",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)

    def _rv(v) -> str:
        if not isinstance(v, dict) or not v:
            return ""
        return " ".join(f"{k[:3]}={float(val):.2f}" for k, val in v.items())

    out = []
    for r in rows:
        row = {c: r.get(c) for c in cols}
        row["reward_vector"] = _rv(r.get("reward_vector"))
        out.append(row)
    return pd.DataFrame(out)


# ── Battle Viewer ─────────────────────────────────────────────────────────────
# Single-command playback browser: filter run → model → scenario, step
# the battle turn-by-turn, and compare two models head-to-head on the
# same scenario+seed.

PLAYBACK_ROOT = Path(
    os.environ.get(
        "OPENRA_BENCH_PLAYBACK_ROOT", Path(__file__).parent / "playback"
    )
)


def _bv_scan():
    try:
        from openra_bench.battle_viewer import scan

        return scan(PLAYBACK_ROOT)
    except Exception:  # noqa: BLE001 — empty/missing root → empty viewer
        return []


def _bv_turn_md(v: dict, heading: str) -> str:
    if not v or v.get("n_turns", 0) == 0:
        return f"### {heading}\n\n_no episode / no turns_"
    m = v.get("manifest", {})
    g = v.get("goal", {}) or {}
    lines = [
        f"### {heading}",
        f"**{m.get('model','?')}** · run `{m.get('run_id','?')}` · "
        f"{m.get('scenario','?')} · seed {m.get('seed','?')} · "
        f"outcome **{m.get('outcome','?')}**",
        f"**turn {v.get('turn')} / {v['n_turns']}** · tick "
        f"{v.get('tick')}"
        + (f" · ⚡ {v['interrupt']}" if v.get("interrupt") else ""),
    ]
    if g:
        parts = []
        for leaf in g.get("leaves", []):
            mark = (
                "✅" if leaf.get("satisfied")
                else f"{float(leaf.get('ratio', 0.0)):.0%}"
            )
            parts.append(
                f"{leaf['name']} {leaf.get('current')}/"
                f"{leaf.get('target')} {mark}"
            )
        bars = " · ".join(parts)
        rv = g.get("reward_vector", {})
        lines += [
            f"**objective: {g.get('objective_progress',0):.0%}**"
            + ("  ✅ WON" if g.get("won") else ""),
            (f"_{bars}_" if bars else ""),
            "reward vector: "
            + " ".join(f"`{k}={float(x):.2f}`" for k, x in rv.items()),
        ]
    # System prompt (the deterministic scenario knowledge the model
    # was given) — collapsible, shown with char count like the
    # training pipeline viewer.
    sp = str(v.get("system_prompt") or "")
    if sp:
        lines += [
            "", f"<details><summary>🧠 system prompt ({len(sp)} chars)"
            "</summary>\n\n```\n" + sp[:6000] + "\n```\n</details>"
        ]
    # DEBRIEF / briefing actually handed to the model this turn
    # (post-interrupt this is the scoped re-prompt).
    deb = str(v.get("debrief") or v.get("briefing") or "")
    if deb:
        tag = "⚡ DEBRIEF (interrupt)" if v.get("interrupt") else "briefing"
        lines += [
            "", f"<details open><summary>{tag}</summary>\n\n```\n"
            + deb[:8000] + "\n```\n</details>"
        ]
    if v.get("reasoning"):
        lines += ["", "**🤔 reasoning (thinking)**",
                  "> " + str(v["reasoning"]).replace("\n", "\n> ")]
    if v.get("assistant_text"):
        lines += ["", "**model said**", str(v["assistant_text"])]
    cmds = v.get("commands", [])
    lines += ["", "**tool calls**", "```\n" + (
        "\n".join(cmds) if cmds else "(none)") + "\n```"]
    if v.get("tool_result"):
        lines += [f"tool result: `{v['tool_result']}`"]
    sig = v.get("signals", {})
    if sig:
        lines += ["signals: " + " ".join(
            f"`{k}={sig[k]}`" for k in sig)]
    return "\n\n".join(s for s in lines if s != "")


def _bv_b_label(e) -> str:
    return f"{e.run_id} / {e.model} ({e.outcome})"


def bv_runs():
    from openra_bench.battle_viewer import runs

    idx = _bv_scan()
    rs = runs(idx)
    return idx, gr.update(choices=rs, value=rs[0] if rs else None)


def bv_on_run(idx, run):
    from openra_bench.battle_viewer import models

    ms = models(idx or [], run) if run else []
    return gr.update(choices=ms, value=ms[0] if ms else None)


def bv_on_model(idx, run, model):
    from openra_bench.battle_viewer import scenarios

    sc = scenarios(idx or [], run, model) if (run and model) else []
    return gr.update(choices=sc, value=sc[0] if sc else None)


def _bv_render(idx, run, model, scen, turn, compare, b_choice):
    from openra_bench.battle_viewer import (
        compare_candidates,
        episode_view,
        find,
    )

    idx = idx or []
    a = find(idx, run, model, scen) if (run and model and scen) else None
    if a is None:
        return (None, _bv_turn_md({}, "A"), None,
                _bv_turn_md({}, "B"), "—", gr.update())
    av = episode_view(a.dir, turn)
    n = av.get("n_turns", 1)
    ti = av.get("turn_idx", 0)
    cands = compare_candidates(idx, a)
    labels = [_bv_b_label(e) for e in cands]
    bv = {}
    if compare and b_choice:
        by = {_bv_b_label(e): e for e in cands}
        be = by.get(b_choice)
        if be is not None:
            bv = episode_view(be.dir, turn)
    return (
        av.get("minimap_png"),
        _bv_turn_md(av, "A"),
        bv.get("minimap_png") if compare else None,
        _bv_turn_md(bv, "B") if compare else "_comparison off_",
        f"turn {ti + 1} / {n}",
        gr.update(choices=labels,
                  value=b_choice if b_choice in labels else (
                      labels[0] if labels else None)),
    )


# ── Filtering ─────────────────────────────────────────────────────────────────


def filter_leaderboard(
    search: str,
    agent_types: list[str],
    opponent: str,
    show_unverified: bool = True,
) -> pd.DataFrame:
    """Filter leaderboard by search, agent type, opponent, and verification status."""
    df = load_data()

    # Filter by verification status
    if not show_unverified:
        df = df[df["Status"].str.contains("Verified</span>", na=False)
               & ~df["Status"].str.contains("Unverified", na=False)]

    # Filter by agent type
    if agent_types:
        df = df[df["Type"].isin(agent_types)]

    # Filter by opponent
    if opponent and opponent != "All":
        df = df[df["Opponent"] == opponent]

    # Search by agent name (regex with fallback to literal on invalid patterns)
    if search and search.strip():
        patterns = [p.strip() for p in search.split(",") if p.strip()]
        mask = pd.Series([False] * len(df), index=df.index)
        for pattern in patterns:
            try:
                mask |= df["Agent"].str.contains(pattern, case=False, regex=True, na=False)
            except re.error:
                mask |= df["Agent"].str.contains(
                    re.escape(pattern), case=False, regex=True, na=False
                )
        df = df[mask]

    # Re-rank after filtering
    df = df.reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    return add_type_badges(df)


# ── Result Persistence ────────────────────────────────────────────────────────

SUBMISSIONS_DIR = Path(__file__).parent / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

GAMES_JSONL = SUBMISSIONS_DIR / "games.jsonl"
MIN_GAMES_FOR_LEADERBOARD = 5

# CommitScheduler pushes submissions to HF dataset (only on HF Spaces)
_scheduler = None
if os.environ.get("HF_TOKEN") and os.environ.get("SPACE_ID"):
    try:
        from huggingface_hub import CommitScheduler

        _scheduler = CommitScheduler(
            repo_id="openra-rl/bench-results",
            repo_type="dataset",
            folder_path=str(SUBMISSIONS_DIR),
            every=5,
            token=os.environ["HF_TOKEN"],
        )
    except Exception:
        pass  # Running locally without HF token — skip


def _sanitize_csv_value(val):
    """Strip leading characters that trigger formula execution in spreadsheets."""
    if isinstance(val, str):
        while val and val[0] in ("=", "+", "-", "@", "\t", "\r", "\n"):
            val = val[1:]
        val = val.replace("\n", " ").replace("\r", " ")
    return val


# ── Rate Limiting ────────────────────────────────────────────────────────────

_submit_times: dict[str, list[float]] = defaultdict(list)
MAX_SUBMITS_PER_HOUR = 20


def _check_rate_limit(identifier: str = "global") -> tuple[bool, str]:
    """Simple in-memory rate limiter. Returns (allowed, error_message)."""
    now = time.time()
    times = _submit_times[identifier]
    _submit_times[identifier] = [t for t in times if now - t < 3600]
    if len(_submit_times[identifier]) >= MAX_SUBMITS_PER_HOUR:
        return False, "Rate limit exceeded (max 20 submissions per hour). Try again later."
    _submit_times[identifier].append(now)
    return True, ""


# ── HF Identity Verification ─────────────────────────────────────────────────


def _verify_hf_token(token: str) -> tuple[str, str]:
    """Verify a HuggingFace token and return the username.

    Returns (hf_username, error_message).
    On success: ("username", "").
    On failure: ("", "reason").
    """
    if not token or not token.strip():
        return "", "no token provided"
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.whoami(token=token.strip())
        username = info.get("name", "")
        if not username:
            return "", "token valid but no username found"
        return username, ""
    except Exception as e:
        logger.debug("HF token verification failed: %s", e)
        return "", f"invalid token: {e}"


# ── Raw Game Storage & Aggregation ────────────────────────────────────────────


def _save_raw_game(data: dict) -> None:
    """Append a single game result to the raw games log.

    Strips ``hf_token`` before writing (only ``hf_username`` is persisted).
    """
    safe = {k: v for k, v in data.items() if k != "hf_token"}
    with open(GAMES_JSONL, "a") as f:
        f.write(json.dumps(safe) + "\n")
    # Also save to results.jsonl for CommitScheduler → HF dataset
    jsonl_path = SUBMISSIONS_DIR / "results.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(safe) + "\n")


def _load_raw_games() -> list[dict]:
    """Load all raw games from games.jsonl."""
    if not GAMES_JSONL.exists():
        return []
    games = []
    for line in GAMES_JSONL.read_text().splitlines():
        if line.strip():
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return games


def _aggregate_agent_games(
    agent_name: str, agent_type: str, opponent: str,
    all_games: list[dict] | None = None,
    hf_username: str = "",
) -> tuple[int, dict | None]:
    """Aggregate all games for a specific agent+opponent pair.

    When *hf_username* is non-empty, only games with a matching
    ``hf_username`` are included. Anonymous games (empty hf_username)
    are never aggregated.

    Returns (game_count, aggregated_row_or_None).
    aggregated_row is None if game_count < MIN_GAMES_FOR_LEADERBOARD.
    """
    if all_games is None:
        all_games = _load_raw_games()

    if not hf_username:
        # Anonymous games are not aggregated
        return 0, None

    matching = [
        g for g in all_games
        if g.get("agent_name") == agent_name
        and g.get("agent_type") == agent_type
        and g.get("opponent") == opponent
        and g.get("hf_username") == hf_username
    ]
    count = len(matching)
    if count < MIN_GAMES_FOR_LEADERBOARD:
        return count, None

    game_results = []
    for g in matching:
        game_results.append({
            "win": g.get("win", g.get("result") == "win"),
            "kills_cost": g.get("kills_cost", 0),
            "deaths_cost": g.get("deaths_cost", 0),
            "assets_value": g.get("assets_value", 0),
            "ticks": g.get("ticks", 0),
        })

    raw_score = compute_composite_score(game_results)
    multiplier = DIFFICULTY_MULTIPLIER.get(opponent, 1.0)

    total_kills = sum(g["kills_cost"] for g in game_results)
    total_deaths = sum(g["deaths_cost"] for g in game_results)

    return count, {
        "agent_name": agent_name,
        "agent_type": agent_type,
        "opponent": opponent,
        "difficulty": opponent,
        "games": count,
        "win_rate": round(100.0 * sum(1 for g in game_results if g["win"]) / count, 1),
        "score": round(raw_score * multiplier, 1),
        "avg_kills": round(total_kills / count),
        "avg_deaths": round(total_deaths / count),
        "kd_ratio": round(total_kills / max(total_deaths, 1), 2),
        "avg_economy": round(sum(g["assets_value"] for g in game_results) / count),
        "avg_game_length": round(sum(g["ticks"] for g in game_results) / count),
        "timestamp": max((g.get("timestamp", "")[:10] for g in matching), default=""),
        "replay_url": next(
            (g.get("replay_url", "") for g in reversed(matching) if g.get("replay_url")),
            "",
        ),
        "agent_url": next(
            (g.get("agent_url", "") for g in reversed(matching) if g.get("agent_url")),
            "",
        ),
        "hf_username": hf_username,
        "verified": True,
    }


def _single_game_row(game: dict) -> dict:
    """Build a leaderboard row from a single anonymous game."""
    game_results = [{
        "win": game.get("win", game.get("result") == "win"),
        "kills_cost": game.get("kills_cost", 0),
        "deaths_cost": game.get("deaths_cost", 0),
        "assets_value": game.get("assets_value", 0),
        "ticks": game.get("ticks", 0),
    }]
    raw_score = compute_composite_score(game_results)
    opponent = game.get("opponent", "Normal")
    multiplier = DIFFICULTY_MULTIPLIER.get(opponent, 1.0)
    kills = game.get("kills_cost", 0)
    deaths = game.get("deaths_cost", 0)

    return {
        "agent_name": game.get("agent_name", ""),
        "agent_type": game.get("agent_type", ""),
        "opponent": opponent,
        "difficulty": opponent,
        "games": 1,
        "win_rate": round(100.0 * int(game_results[0]["win"]), 1),
        "score": round(raw_score * multiplier, 1),
        "avg_kills": kills,
        "avg_deaths": deaths,
        "kd_ratio": round(kills / max(deaths, 1), 2),
        "avg_economy": game.get("assets_value", 0),
        "avg_game_length": game.get("ticks", 0),
        "timestamp": game.get("timestamp", "")[:10],
        "replay_url": game.get("replay_url", ""),
        "agent_url": game.get("agent_url", ""),
        "hf_username": "",
        "verified": False,
    }


def _rebuild_leaderboard() -> None:
    """Rebuild leaderboard CSV from raw games.

    Verified users (non-empty hf_username) are aggregated by
    (hf_username, agent_name, agent_type, opponent) with a minimum of
    5 games to appear. Anonymous games (empty hf_username) appear as
    individual rows marked as unverified.
    """
    all_games = _load_raw_games()
    if not all_games:
        return  # No games yet, keep existing CSV as-is

    rows = []

    # 1. Aggregate verified games
    verified_groups = set()
    for g in all_games:
        hf_user = g.get("hf_username", "")
        if hf_user:
            key = (hf_user, g.get("agent_name", ""), g.get("agent_type", ""), g.get("opponent", ""))
            verified_groups.add(key)

    for hf_user, name, atype, opp in verified_groups:
        count, agg = _aggregate_agent_games(name, atype, opp, all_games, hf_username=hf_user)
        if agg is not None:
            rows.append(agg)

    # 2. Add anonymous games as individual rows
    for g in all_games:
        if not g.get("hf_username"):
            rows.append(_single_game_row(g))

    if not rows:
        return  # No qualifying entries

    rows.sort(key=lambda r: r.get("score", 0), reverse=True)

    fieldnames = LEADERBOARD_FIELDNAMES
    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _sanitize_csv_value(row.get(k, "")) for k in fieldnames})


LEADERBOARD_FIELDNAMES = [
    "agent_name", "agent_type", "opponent", "difficulty", "games",
    "win_rate", "score", "avg_kills", "avg_deaths", "kd_ratio",
    "avg_economy", "avg_game_length", "timestamp", "replay_url", "agent_url",
    "hf_username", "verified",
]


# ── Submission Handling ───────────────────────────────────────────────────────

MAX_REPLAY_SIZE = 10 * 1024 * 1024  # 10 MB

VALID_OPPONENTS = {"Beginner", "Easy", "Medium", "Normal", "Hard"}
VALID_AGENT_TYPES = {"Scripted", "LLM", "RL"}
REQUIRED_FIELDS = [
    "agent_name", "agent_type", "opponent", "result",
    "ticks", "kills_cost", "deaths_cost", "assets_value",
]


def validate_submission(data: dict) -> tuple[bool, str]:
    """Validate an uploaded JSON submission.

    Returns (is_valid, error_message).
    """
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: {field}"

    if data["agent_type"] not in VALID_AGENT_TYPES:
        return False, (
            f"Invalid agent_type: {data['agent_type']}. "
            f"Must be one of: {', '.join(sorted(VALID_AGENT_TYPES))}"
        )

    if data["opponent"] not in VALID_OPPONENTS:
        return False, (
            f"Invalid opponent: {data['opponent']}. "
            f"Must be one of: {', '.join(sorted(VALID_OPPONENTS))}"
        )

    # Type checks for numeric fields
    for field in ("ticks", "kills_cost", "deaths_cost", "assets_value"):
        if not isinstance(data[field], (int, float)):
            return False, f"Field '{field}' must be a number"

    # String length limits
    if len(str(data["agent_name"])) > 100:
        return False, "agent_name must be 100 characters or fewer"

    # agent_url: optional, but must be http(s) if provided
    agent_url = str(data.get("agent_url", "")).strip()
    if agent_url and not agent_url.startswith(("http://", "https://")):
        return False, "agent_url must be an HTTP(S) URL"
    if len(agent_url) > 500:
        return False, "agent_url must be 500 characters or fewer"

    return True, ""


def handle_upload(json_file, replay_file) -> tuple[str, pd.DataFrame]:
    """Process an uploaded bench submission JSON + optional replay."""
    if json_file is None:
        return "Please upload a JSON file.", add_type_badges(load_data())

    allowed, err = _check_rate_limit()
    if not allowed:
        return err, add_type_badges(load_data())

    try:
        with open(json_file.name) as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        return f"Invalid JSON: {e}", add_type_badges(load_data())

    is_valid, error = validate_submission(data)
    if not is_valid:
        return f"Validation error: {error}", add_type_badges(load_data())

    hf_username, anon_warning = _process_identity(data)

    # Save replay if provided
    if replay_file is not None:
        import shutil

        orig = Path(replay_file.name)
        if orig.stat().st_size > MAX_REPLAY_SIZE:
            return "Replay file too large (max 10 MB).", add_type_badges(load_data())
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "", data["agent_name"].replace("/", "_").replace(" ", "_"))[:30]
        replay_name = f"replay-{slug}-{ts}.orarep"
        shutil.copy2(str(orig), SUBMISSIONS_DIR / replay_name)
        data["replay_url"] = replay_name

    _save_raw_game(data)
    _rebuild_leaderboard()

    agent_name = data["agent_name"]
    opponent = data["opponent"]

    if not hf_username:
        msg = (
            f"Recorded anonymous game for **{agent_name}** vs {opponent}. "
            f"Add an HF token to aggregate games and track progress."
        )
        if anon_warning:
            msg = f"{anon_warning} {msg}"
    else:
        count, agg = _aggregate_agent_games(
            agent_name, data["agent_type"], opponent, hf_username=hf_username,
        )
        if count < MIN_GAMES_FOR_LEADERBOARD:
            remaining = MIN_GAMES_FOR_LEADERBOARD - count
            msg = (
                f"Recorded game {count}/{MIN_GAMES_FOR_LEADERBOARD} for "
                f"**{agent_name}** vs {opponent}. "
                f"Play {remaining} more game{'s' if remaining != 1 else ''} "
                f"to appear on the leaderboard!"
            )
        else:
            msg = (
                f"**{agent_name}** vs {opponent} updated \u2014 "
                f"{count} games, score **{agg['score']}** (win rate {agg['win_rate']}%)"
            )

    return msg, add_type_badges(load_data())


def _process_identity(data: dict) -> tuple[str, str]:
    """Verify HF token if present, set hf_username on data.

    Returns (hf_username, warning_message).
    """
    token = data.pop("hf_token", "")
    if token:
        hf_username, err = _verify_hf_token(token)
        if hf_username:
            data["hf_username"] = hf_username
            return hf_username, ""
        else:
            data["hf_username"] = ""
            return "", f"HF token verification failed ({err}). Submitted as anonymous."
    data.setdefault("hf_username", "")
    return "", ""


def _build_response(agent_name: str, agent_type: str, opponent: str,
                    hf_username: str, anonymous_warning: str,
                    all_games: list[dict] | None = None) -> str:
    """Build a response message after saving a game."""
    parts = []
    if anonymous_warning:
        parts.append(anonymous_warning)

    if not hf_username:
        # Anonymous: not aggregated
        parts.append(
            f"OK: recorded anonymous game for {agent_name} vs {opponent}. "
            f"Add an HF token to aggregate games and track progress."
        )
        return " ".join(parts)

    count, agg = _aggregate_agent_games(
        agent_name, agent_type, opponent,
        all_games=all_games, hf_username=hf_username,
    )

    if count < MIN_GAMES_FOR_LEADERBOARD:
        remaining = MIN_GAMES_FOR_LEADERBOARD - count
        parts.append(
            f"OK: recorded game {count}/{MIN_GAMES_FOR_LEADERBOARD} for "
            f"{agent_name} vs {opponent}. "
            f"Play {remaining} more game{'s' if remaining != 1 else ''} "
            f"to appear on the leaderboard!"
        )
    else:
        parts.append(
            f"OK: {agent_name} vs {opponent} updated \u2014 "
            f"{count} games, score {agg['score']} (win rate {agg['win_rate']}%)"
        )
    return " ".join(parts)


def handle_api_submit(json_data: str) -> str:
    """API endpoint: accept JSON string submission. Used by CLI auto-upload."""
    allowed, err = _check_rate_limit()
    if not allowed:
        return err

    try:
        data = json.loads(json_data)
    except (json.JSONDecodeError, Exception) as e:
        return f"Invalid JSON: {e}"

    is_valid, error = validate_submission(data)
    if not is_valid:
        return f"Validation error: {error}"

    hf_username, anon_warning = _process_identity(data)

    _save_raw_game(data)
    _rebuild_leaderboard()

    return _build_response(
        data["agent_name"], data["agent_type"], data["opponent"],
        hf_username, anon_warning,
    )


def handle_api_submit_with_replay(json_data: str, replay_file) -> str:
    """API endpoint: accept JSON + replay file. Used by CLI with --replay."""
    allowed, err = _check_rate_limit()
    if not allowed:
        return err

    try:
        data = json.loads(json_data)
    except (json.JSONDecodeError, Exception) as e:
        return f"Invalid JSON: {e}"

    is_valid, error = validate_submission(data)
    if not is_valid:
        return f"Validation error: {error}"

    hf_username, anon_warning = _process_identity(data)

    # Save replay if provided
    if replay_file is not None:
        import shutil

        orig = Path(replay_file) if isinstance(replay_file, str) else Path(replay_file.name)
        if orig.exists() and orig.stat().st_size > MAX_REPLAY_SIZE:
            return "Replay file too large (max 10 MB)"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "", data["agent_name"].replace("/", "_").replace(" ", "_"))[:30]
        replay_name = f"replay-{slug}-{ts}.orarep"
        shutil.copy2(str(orig), SUBMISSIONS_DIR / replay_name)
        data["replay_url"] = replay_name

    _save_raw_game(data)
    _rebuild_leaderboard()

    return _build_response(
        data["agent_name"], data["agent_type"], data["opponent"],
        hf_username, anon_warning,
    )


# ── UI ────────────────────────────────────────────────────────────────────────

ABOUT_MD = """
## What is OpenRA-Bench?

**OpenRA-Bench** is a standardized benchmark for evaluating AI agents playing
[Red Alert](https://www.openra.net/) through the
[OpenRA-RL](https://openra-rl.dev) environment.

### Evaluation Protocol

- **Game**: Red Alert (OpenRA engine)
- **Format**: 1v1 agent vs built-in AI
- **Opponents**: Beginner, Easy, Medium, Normal, Hard difficulty
- **Games per entry**: Minimum 5 games per configuration
- **Metrics**: Win rate, composite score, K/D ratio, economy

### Composite Score

The benchmark score combines four components, scaled by opponent difficulty:

| Component | Weight | Description |
|-----------|--------|-------------|
| Win Rate | 50% | Percentage of games won |
| Military Efficiency | 20% | Kill/death cost ratio (0 if no combat) |
| Economy | 20% | Final asset value (normalized) |
| Speed | 10% | Faster decisive games score higher |

**Difficulty multiplier**: Beginner (0.5x), Easy (0.7x), Medium (0.85x), Normal (1.0x), Hard (1.2x)

**Minimum games**: 5 games required per agent+opponent to appear on the leaderboard (verified users only).

### Identity & Verification

- **Verified**: Include your HuggingFace token (`hf_token`) in submissions.
  Games are aggregated by HF username + agent name + opponent.
- **Anonymous**: No token required. Games appear individually with an
  "Unverified" badge and are not aggregated across sessions.

### Agent Types

- **Scripted**: Rule-based bots with hardcoded strategies
- **LLM**: Language model agents (Claude, GPT, etc.)
- **RL**: Reinforcement learning policies (PPO, SAC, etc.)

### Links

- [OpenRA-RL Documentation](https://openra-rl.dev)
- [GitHub Repository](https://github.com/yxc20089/OpenRA-RL)
- [OpenRA-Bench Source](https://github.com/yxc20089/OpenRA-Bench)
- [OpenEnv Framework](https://huggingface.co/openenv)
- [HuggingFace Space](https://huggingface.co/spaces/openra-rl/OpenRA-Bench)
"""

SUBMIT_MD = """
---

## Other Submission Methods

### CLI Auto-Upload

Set `BENCH_URL` and optionally `HF_TOKEN` in your OpenRA-RL config. Results
upload automatically after each game. With a HF token, games are aggregated
under your verified username:

```yaml
# config.yaml
agent:
  bench_url: "https://openra-rl-openra-bench.hf.space"
  hf_token: "hf_..."  # Optional: enables verified aggregation
```

### CLI Manual Upload

Upload a previously exported bench JSON:

```bash
python -m openra_env.bench_submit ~/.openra-rl/bench-exports/bench-*.json
```

### Batch Evaluation (5+ games)

```bash
git clone https://github.com/yxc20089/OpenRA-Bench.git
cd OpenRA-Bench
pip install -r requirements.txt
pip install openra-rl openra-rl-util

python evaluate.py \\
    --agent scripted \\
    --agent-name "MyBot-v1" \\
    --agent-type Scripted \\
    --opponent Normal \\
    --games 10 \\
    --server http://localhost:8000
```

### Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `--agent` | Agent type: `scripted`, `llm`, `mcp`, `custom` |
| `--agent-name` | Display name on the leaderboard |
| `--agent-type` | Category: `Scripted`, `LLM`, `RL` |
| `--opponent` | AI difficulty: `Beginner`, `Easy`, `Medium`, `Normal`, `Hard` |
| `--games` | Number of games (minimum 5) |
| `--server` | OpenRA-RL server URL (local or HuggingFace-hosted) |

### Custom Agents

Implement the standard `reset/step` loop:

```python
from openra_env.client import OpenRAEnv
from openra_env.models import OpenRAAction

async with OpenRAEnv("http://localhost:8000") as env:
    obs = await env.reset()
    while not obs.done:
        action = your_agent.decide(obs)
        obs = await env.step(action)
```

Then run `evaluate.py --agent custom` with your agent integrated.
"""


# ── Scenarios tab (interactive catalog) ────────────────────────────────────────

_CAP_COLORS = {
    "perception": "#7497db",
    "reasoning": "#9b8cce",
    "action": "#5fae7a",
    "adversarial": "#d2683c",
}

_translate_cache: dict[str, str] = {}


def _google_translate_zh(text: str) -> str:
    """Translate English text to Simplified Chinese via Google Translate."""
    if not text or not text.strip():
        return text
    if text in _translate_cache:
        return _translate_cache[text]
    import urllib.parse
    import urllib.request

    url = (
        "https://translate.googleapis.com/translate_a/single"
        "?client=gtx&sl=en&tl=zh-CN&dt=t&q="
        + urllib.parse.quote(text)
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            result = "".join(seg[0] for seg in data[0] if seg[0])
        _fixups = [
            ("游戏前勾选", "游戏刻"), ("游戏勾选", "游戏刻"),
            ("游戏滴答", "游戏刻"), ("游戏刻度", "游戏刻"),
            ("游戏蜱虫", "游戏刻"), ("游戏壁虱", "游戏刻"),
            ("游戏报价", "游戏刻"), ("游戏打勾", "游戏刻"),
            ("决策轮次", "决策回合"), ("决策转弯", "决策回合"),
            ("勾号", "刻"),
        ]
        for wrong, right in _fixups:
            result = result.replace(wrong, right)
        _translate_cache[text] = result
        return result
    except Exception:
        return text


def _scenarios_catalog_df() -> pd.DataFrame:
    """Load every active scenario pack into a DataFrame for the catalog."""
    try:
        from openra_bench.scenarios import discover_packs
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=[
            "ID", "Title", "Capability", "Map", "Real-World Meaning",
            "Robotics Analogue", "Benchmark Anchor",
        ])
    rows = []
    for p in discover_packs():
        if p.meta.status != "active":
            continue
        anchors = ", ".join(p.meta.benchmark_anchor) if p.meta.benchmark_anchor else ""
        rows.append({
            "ID": p.meta.id,
            "Title": p.meta.title,
            "Capability": p.meta.capability,
            "Map": p.base_map if isinstance(p.base_map, str) else "generated",
            "Real-World Meaning": p.meta.real_world_meaning,
            "Robotics Analogue": p.meta.robotics_analogue,
            "Benchmark Anchor": anchors,
        })
    return pd.DataFrame(rows)


def _scenarios_filter(search: str, capabilities: list[str]) -> pd.DataFrame:
    """Filter the scenario catalog by search term and capability."""
    df = _scenarios_catalog_df()
    if not len(df):
        return df
    df = df[df["Capability"].isin(capabilities or [])]
    if search and search.strip():
        q = search.strip().lower()
        mask = (
            df["ID"].str.lower().str.contains(q, na=False)
            | df["Title"].str.lower().str.contains(q, na=False)
            | df["Real-World Meaning"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]
    return df.reset_index(drop=True)


def _scenarios_detail_md(pack_id: str, lang: str = "en") -> str:
    """Render full detail for one scenario pack as Markdown.

    When lang='zh', all objectives are shown in Chinese via Google Translate.
    """
    if not pack_id or not pack_id.strip():
        return "_Select a scenario from the table above to see details._"
    pack_id = pack_id.strip()
    try:
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import load_pack
        from openra_bench.scenarios.loader import PACKS_DIR

        path = PACKS_DIR / f"{pack_id}.yaml"
        if not path.exists():
            return f"Pack `{pack_id}` not found."
        pack = load_pack(path)
    except Exception as e:  # noqa: BLE001
        return f"Error loading `{pack_id}`: {e}"

    cap = pack.meta.capability
    color = _CAP_COLORS.get(cap, "#666")
    anchors = ", ".join(pack.meta.benchmark_anchor) if pack.meta.benchmark_anchor else "none"

    rwm = pack.meta.real_world_meaning
    rob = pack.meta.robotics_analogue
    if lang == "zh":
        rwm = _google_translate_zh(rwm)
        rob = _google_translate_zh(rob)

    why_label = "为什么有这个场景：" if lang == "zh" else "Why this exists:"
    robo_label = "机器人类比：" if lang == "zh" else "Robotics analogue:"
    anchor_label = "基准锚点：" if lang == "zh" else "Benchmark anchors:"
    levels_label = "### 难度等级" if lang == "zh" else "### Levels"

    lines = [
        f"## {_md_escape(pack.meta.title)}",
        f"**ID:** `{pack.meta.id}` | **Capability:** "
        f"<span style='background:{color};color:#fff;padding:2px 8px;"
        f"border-radius:4px;font-size:0.85em'>{cap}</span> | "
        f"**Map:** `{pack.base_map if isinstance(pack.base_map, str) else 'generated'}`",
        "",
        f"**{why_label}** {_md_escape(rwm)}",
        "",
        f"**{robo_label}** {_md_escape(rob)}",
        "",
        f"**{anchor_label}** {_md_escape(anchors)}",
        "",
        "---",
        "",
        levels_label,
    ]

    cells = []
    if pack.configs:
        for c in pack.configs:
            try:
                cl = pack.compile_config(c.name)
                cells.append((c.name, cl))
            except Exception as e:  # noqa: BLE001
                cells.append((c.name, e))
    else:
        for lv in ("easy", "medium", "hard"):
            try:
                cl = pack.compile(lv)
                cells.append((lv, cl))
            except Exception as e:  # noqa: BLE001
                cells.append((lv, e))

    diff_zh = {"easy": "简单", "medium": "中等", "hard": "困难"}

    for label, cl in cells:
        if isinstance(cl, Exception):
            lines.append(f"\n**{label}** — compile error: {cl}")
            continue
        fog = getattr(cl, "fog_mode", "vision")
        cash_str = f" | cash: {cl.starting_cash}" if cl.starting_cash is not None else ""
        display_label = diff_zh.get(label, label) if lang == "zh" else label
        lines.append(
            f"\n**{display_label}** (level {cl.level} | fog: {fog} | "
            f"turns: {cl.max_turns}{cash_str})"
        )
        try:
            ob = objective_brief(
                cl.scenario.description, cl.win_condition,
                cl.fail_condition, cl.max_turns,
                getattr(cl, "objective_coords", "exact"),
            )
            if lang == "zh":
                ob = _google_translate_zh(ob)
            lines.append(f"```\n{ob}\n```")
        except Exception as e:  # noqa: BLE001
            lines.append(f"_(objective error: {e})_")

    return "\n".join(lines)


# ── Play tab (human-labeling machine) ─────────────────────────────────────────
# Lets a human play the exact scenarios LLM agents are scored on, by
# clicking the minimap — the Phase 2 human-labeling machine. Backed by
# openra_bench.human_labeling.InteractiveSession (turn-steppable) so a
# human's run is scored by the identical rules as a model's.

_PLAY_LEVELS = ["easy", "medium", "hard"]
_PLAY_UPSCALE = 5  # tactical-minimap cell scale for the Play tab

_PLAY_KEYBOARD_JS = r"""
() => {
  if (window.__openraBenchPlayEnterBound) return;
  window.__openraBenchPlayEnterBound = true;
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.metaKey ||
        event.ctrlKey || event.altKey || event.repeat) {
      return;
    }
    const target = event.target;
    const tag = (target && target.tagName || "").toLowerCase();
    if (["input", "textarea", "select", "button"].includes(tag) ||
        (target && target.isContentEditable)) {
      return;
    }
    const root = document.getElementById("play-end-turn-btn");
    if (!root) return;
    const rect = root.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return;
    const button = root.tagName && root.tagName.toLowerCase() === "button"
      ? root : root.querySelector("button");
    if (!button || button.disabled) return;
    event.preventDefault();
    button.click();
  }, true);
}
"""


def _play_scenarios() -> list[str]:
    """Active pack ids playable in the Play tab."""
    try:
        from openra_bench.scenarios import load_pack
        from openra_bench.scenarios.loader import PACKS_DIR

        out = []
        for f in sorted(PACKS_DIR.glob("*.yaml")):
            if f.name.startswith(("_", "TEMPLATE")):
                continue
            try:
                if load_pack(f).meta.status == "active":
                    out.append(f.stem)
            except Exception:  # noqa: BLE001
                continue
        return out
    except Exception:  # noqa: BLE001
        return []


def _md_escape(text: str) -> str:
    """Escape Markdown-significant characters so scenario prose renders
    literally — e.g. the `~` in 'NE ~110,6' must not become strikethrough."""
    text = text.replace("\\", "\\\\")
    for ch in ("~", "*", "_", "`", "#"):
        text = text.replace(ch, "\\" + ch)
    return text


def _play_minimap(render_state: dict, sel=None, queue=None):
    """The Play-tab minimap — `render_tactical_minimap` plus a white
    boundary on the selected units and movement arrows. An arrow points
    to a unit's queued destination this turn; if it has no queued order
    but is already moving, the arrow points to its in-engine target."""
    try:
        from openra_bench.minimap import render_tactical_minimap

        sel_ids = {str(s) for s in (sel or [])}
        # A queued move/attack this turn overrides the in-flight target.
        queued_dest: dict = {}
        for a in queue or []:
            if getattr(a, "mode", "") in (
                "move", "attack", "attack_move"
            ) and getattr(a, "target", None):
                for uid in a.units:
                    queued_dest[str(uid)] = (a.target[0], a.target[1])
        arrows = []
        for u in render_state.get("units_summary", []) or []:
            if not isinstance(u, dict):
                continue
            uid = str(u.get("id", ""))
            fx, fy = u.get("cell_x"), u.get("cell_y")
            if fx is None or fy is None:
                continue
            if uid in queued_dest:
                tx, ty = queued_dest[uid]
                arrows.append((fx, fy, tx, ty, "queued"))
            elif u.get("activity") == "moving" and (
                u.get("target_x") is not None
            ):
                arrows.append(
                    (fx, fy, u["target_x"], u["target_y"], "enroute")
                )
        return render_tactical_minimap(
            render_state, scale=_PLAY_UPSCALE, grid=True, legend=True,
            selected=sel_ids, arrows=arrows,
        )
    except Exception:  # noqa: BLE001
        return None


def _play_render_state(sess, show_objectives: bool = False) -> dict:
    rs = sess.render_state()
    if not show_objectives and "objective_regions" in rs:
        rs = dict(rs)
        rs.pop("objective_regions", None)
    return rs


def _play_objective_md(sess) -> str:
    """The scenario objective — what the human must do to WIN."""
    if sess is None:
        return ""
    obj = (getattr(sess, "objective", "") or "").strip()
    if not obj:
        return ""
    return f"### 🎯 Objective\n{_md_escape(obj)}"


_PLAY_UNIT_COLS = ["sel", "unit", "type", "cell", "hp", "status"]


def _play_units_df(sess, sel):
    """Table of the human's own units. Selected units are marked '▶' and
    sorted to the top so the current selection is obvious. `hp` is the
    0-1 fraction the engine reports, shown as a percentage."""
    if sess is None:
        return pd.DataFrame(columns=_PLAY_UNIT_COLS)
    try:
        rs = sess.render_state()
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=_PLAY_UNIT_COLS)
    selset = {str(s) for s in (sel or [])}
    rows = []
    for u in rs.get("units_summary", []) or []:
        if not isinstance(u, dict):
            continue
        uid = str(u.get("id", ""))
        try:
            hp_txt = f"{int(round(float(u.get('hp', 1.0)) * 100))}%"
        except (TypeError, ValueError):
            hp_txt = "?"
        is_sel = uid in selset
        rows.append({
            "sel": "▶" if is_sel else "",
            "unit": uid,
            "type": u.get("type") or u.get("actor_type") or "?",
            "cell": f"({u.get('cell_x')}, {u.get('cell_y')})",
            "hp": hp_txt,
            "status": u.get("activity", "") or "idle",
            "_sel": is_sel,
        })
    df = pd.DataFrame(rows, columns=_PLAY_UNIT_COLS + ["_sel"])
    # Selected units float to the top so the selection is unmistakable.
    df = (
        df.sort_values("_sel", ascending=False, kind="stable")
        .drop(columns="_sel")
        .reset_index(drop=True)
    )
    return df


def _play_status_md(sess) -> str:
    if sess is None:
        return "_No active session — pick a scenario and click **Start**._"
    st = sess.status()
    line = f"**Turn {st['turn']}/{st['max_turns']}** · tick {st['tick']}"
    if st["done"]:
        line += f" · **{st['outcome'].upper()}** — game over"
        if st.get("save_path"):
            line += (
                f"\n\n_Run saved (standard playback format): "
                f"`{st['save_path']}`_"
            )
    return line


def _play_briefing_md(sess, sel, queue, note: str = "") -> str:
    """The turn panel: current selection + queued orders FIRST, then the
    exact text briefing the model is given for this turn."""
    if sess is None:
        return ""
    try:
        from openra_bench.human_labeling import HumanController

        brief = HumanController._briefing(sess.render_state())
    except Exception:  # noqa: BLE001
        brief = ""
    sel_txt = ", ".join(sel) if sel else "(none)"
    q_txt = "; ".join(a.describe() for a in queue) if queue else "(none)"
    head = f"{note}\n\n" if note else ""
    return (
        f"{head}"
        f"**Selected units:** {sel_txt}  \n"
        f"**Queued this turn:** {q_txt}\n\n"
        f"**Turn briefing — exactly what the model sees:**\n"
        f"```\n{brief}\n```"
    )


def _play_render(sess, sel, queue, show_objectives=False):
    img = (
        _play_minimap(_play_render_state(sess, show_objectives), sel, queue)
        if sess is not None else None
    )
    return (
        img,
        _play_briefing_md(sess, sel, queue),
        _play_status_md(sess),
        _play_units_df(sess, sel),
    )


def _play_start(prev_sess, pack, level, seed, show_objectives=False):
    # Release any prior session's engine env before opening a new one.
    if prev_sess is not None:
        try:
            prev_sess.close()
        except Exception:  # noqa: BLE001
            pass
    empty_units = _play_units_df(None, [])
    if not pack:
        return (
            None, [], [], "", None, "", "_pick a scenario first_",
            empty_units,
        )
    try:
        from openra_bench.human_labeling import InteractiveSession

        sess = InteractiveSession.from_pack(
            pack, level or "easy", int(seed or 1)
        )
    except Exception as e:  # noqa: BLE001
        return (
            None, [], [], "", None, f"⚠️ {e}", "_start failed_",
            empty_units,
        )
    img, brief, status, units = _play_render(sess, [], [], show_objectives)
    return (
        sess, [], [], _play_objective_md(sess),
        img, brief, status, units,
    )


def _play_click(sess, sel, queue, show_objectives, evt: gr.SelectData):
    """Contextual minimap click — classic RTS interaction, no mode:

    * click a cell holding YOUR unit(s)  → select/deselect them (toggle);
    * click an enemy with units selected → queue an ATTACK on it;
    * click empty ground with a selection → queue a MOVE there.

    So 'select a unit, then click where to send it' just works."""
    if sess is None or evt is None or evt.index is None:
        return (
            sel, queue, _play_briefing_md(sess, sel, queue),
            _play_units_df(sess, sel), None,
        )
    note = ""
    try:
        from openra_bench.human_labeling import (
            HumanAction,
            enemy_at_cell,
            minimap_click_to_cell,
            own_units_at_cell,
        )

        px, py = evt.index  # pixels in the displayed (upscaled) image
        rs = sess.render_state()
        rows = [r for r in (rs.get("minimap") or "").split("\n") if r]
        if not rows:
            return (
                sel, queue, _play_briefing_md(sess, sel, queue),
                _play_units_df(sess, sel), None,
            )
        h = len(rows)
        w = max(len(r) for r in rows)
        img_w = w * 6 * _PLAY_UPSCALE
        img_h = h * 6 * _PLAY_UPSCALE
        cx, cy = minimap_click_to_cell(px, py, img_w, img_h, w, h)
        note = f"🖱 Cell **({cx}, {cy})**"

        own_here = own_units_at_cell(rs, cx, cy, radius=0)
        enemy_here = enemy_at_cell(rs, cx, cy, radius=0)
        sel = list(sel)

        if own_here:
            # Toggle-select your own unit(s) on this cell.
            for uid in own_here:
                if uid in sel:
                    sel.remove(uid)
                else:
                    sel.append(uid)
            note += (
                f" — selected unit {', '.join(own_here)} · "
                f"**{len(sel)} selected**"
            )
        elif sel and enemy_here:
            queue = queue + [
                HumanAction(
                    mode="attack", units=list(sel),
                    target_id=enemy_here, target=(cx, cy),
                )
            ]
            note += (
                f" — queued **attack** on enemy {enemy_here} "
                f"({len(sel)} unit(s))"
            )
        elif sel:
            queue = queue + [
                HumanAction(mode="move", units=list(sel), target=(cx, cy))
            ]
            note += f" — queued **move** of {len(sel)} unit(s) here"
        else:
            note += " — empty (select one of your units first)"
    except Exception as e:  # noqa: BLE001
        logger.warning("play click failed: %s", e)
        note = ""
    # Re-render the minimap so the selection boundary + move arrows
    # update live as the player clicks.
    img = (
        _play_minimap(_play_render_state(sess, show_objectives), sel, queue)
        if sess is not None else None
    )
    return (
        sel, queue, _play_briefing_md(sess, sel, queue, note),
        _play_units_df(sess, sel), img,
    )


def _play_end_turn(sess, sel, queue, show_objectives=False):
    if sess is not None and not sess.done:
        try:
            sess.submit_turn(list(queue))
        except Exception as e:  # noqa: BLE001
            logger.warning("play submit_turn failed: %s", e)
    img, brief, status, units = _play_render(sess, [], [], show_objectives)
    return sess, [], [], img, brief, status, units


def _play_clear_queue(sess, sel, show_objectives=False):
    """Cancel queued orders this turn (keeps the unit selection)."""
    img = (
        _play_minimap(_play_render_state(sess, show_objectives), sel, [])
        if sess is not None else None
    )
    return (
        [], _play_briefing_md(sess, sel, []),
        _play_units_df(sess, sel), img,
    )


def _play_clear_selection(sess, queue, show_objectives=False):
    """Cancel the current unit selection (keeps queued orders)."""
    img = (
        _play_minimap(_play_render_state(sess, show_objectives), [], queue)
        if sess is not None else None
    )
    return (
        [], _play_briefing_md(sess, [], queue),
        _play_units_df(sess, []), img,
    )


def _play_toggle_objectives(sess, sel, queue, show_objectives):
    if sess is None:
        return None
    return _play_minimap(
        _play_render_state(sess, show_objectives), sel, queue
    )


def _play_build_item(sess, sel, queue, item, show_objectives=False):
    """Queue a production order for the current turn (Play tab build queue)."""
    note = ""
    if sess is not None:
        item = str(item or "").strip().lower()
        if item:
            try:
                from openra_bench.human_labeling import HumanAction

                queue = queue + [HumanAction(mode="build", unit_type=item)]
                note = f"Queued **build {item}**"
            except Exception as e:  # noqa: BLE001
                logger.warning("play build queue failed: %s", e)
                note = ""
    img = (
        _play_minimap(_play_render_state(sess, show_objectives), sel, queue)
        if sess is not None else None
    )
    return (
        queue, "", _play_briefing_md(sess, sel, queue, note),
        _play_units_df(sess, sel), img,
    )


# ── Human-study mode ─────────────────────────────────────────────────
# Walks a recruited player through the fixed 24-pack study subset under
# 3 conditions (72 games, per-player counterbalanced). Every game saves
# to the standard Playback format — apples-to-apple with model runs.

def _study_progress_md(st: dict) -> str:
    pl = st.get("playlist", [])
    i = st.get("idx", 0)
    if not pl:
        return "_Enter your name and click **Begin study**._"
    if i >= len(pl):
        return (
            f"**✅ Study complete** — all {len(pl)} games done. "
            f"Thank you, `{st['player']}`!"
        )
    pack, level, cond = pl[i]
    return (
        f"**Study — game {i + 1} / {len(pl)}**  ·  player `{st['player']}`\n\n"
        f"`{pack}` [{level}]  ·  condition: **{cond}**  \n"
        f"_Play to game-over, then click **Next scenario ▶**._"
    )


def _study_render(st: dict, prev_sess):
    """Open the study session for st's current cell and render it."""
    if prev_sess is not None:
        try:
            prev_sess.close()
        except Exception:  # noqa: BLE001
            pass
    empty = _play_units_df(None, [])
    pl = st.get("playlist", [])
    if st.get("idx", 0) >= len(pl):
        return (None, [], [], "", None, "", "", empty, st,
                _study_progress_md(st))
    pack, level, cond = pl[st["idx"]]
    try:
        from openra_bench.human_study import open_study_session

        sess = open_study_session(
            pack, level, cond, player=st["player"], seed=1
        )
    except Exception as e:  # noqa: BLE001
        return (None, [], [], "", None, f"⚠️ {e}",
                "_study load failed_", empty, st, _study_progress_md(st))
    img, brief, status, units = _play_render(sess, [], [], False)
    return (sess, [], [], _play_objective_md(sess), img, brief, status,
            units, st, _study_progress_md(st))


def _study_begin(prev_sess, player):
    import hashlib

    from openra_bench.human_study import study_playlist

    player = (player or "").strip() or "anon"
    # Per-player counterbalancing — a stable seed from the name.
    seed = int(hashlib.md5(player.encode()).hexdigest()[:8], 16)
    st = {"player": player, "playlist": study_playlist(seed), "idx": 0}
    return _study_render(st, prev_sess)


def _study_next(prev_sess, st):
    if not st or "playlist" not in st:
        return _study_render({"player": "anon", "playlist": []}, prev_sess)
    st = dict(st)
    st["idx"] = st.get("idx", 0) + 1
    return _study_render(st, prev_sess)


# ── Playlist mode (cold-start non-gamer UX) ──────────────────────────
# A separate, simpler tab from `Play` (sandbox/debug) and the 24-pack
# `Study` accordion (recruited tester). The Playlist tab walks a
# non-gamer through `NOVICE_PLAYLIST` — a curated 20-pack list whose
# objectives are visually obvious and whose tool surface fits the
# reduced palette (move/attack/build/end-turn). Plain-language
# objective above the minimap, jargon-substituted everywhere, auto-
# advance on game-over, progress bar at the top, summary table at the
# end. The underlying engine session is the same `InteractiveSession`
# the `Play` tab uses (and the 24-pack study uses), so a Playlist run
# is still apples-to-apple with a model run on the same scenario.

def _playlist_state(player: str = "anon") -> dict:
    """Initial Playlist tab state — the curated 20-pack list, idx 0,
    no completed games yet, no game-over timestamp."""
    from openra_bench.playlist import NOVICE_PLAYLIST

    return {
        "player": (player or "").strip() or "anon",
        "playlist": list(NOVICE_PLAYLIST),
        "idx": 0,
        "results": [],            # list of session_summary_row dicts
        "done_at": None,          # wall-clock when current game ended
        "submitted": False,       # baseline already posted
    }


def _playlist_progress_md(st: dict) -> str:
    """The progress bar at the top of the Playlist tab — '1 of 20',
    plus the text bar so a non-gamer can see at-a-glance how far
    they've come."""
    from openra_bench.playlist import playlist_progress_bar

    pl = st.get("playlist") or []
    total = len(pl)
    idx = int(st.get("idx", 0) or 0)
    if total == 0:
        return "_Enter your name and click **Start playlist** to begin._"
    if idx >= total:
        return f"### ✅ Playlist complete — {total} of {total}\n\n`{'▮' * 20}  100%`"
    bar = playlist_progress_bar(idx, total)
    return f"### Scenario {idx + 1} of {total}  ·  player `{st.get('player', 'anon')}`\n\n`{bar}`"


def _playlist_objective_md(sess) -> str:
    """The plain-English 1-2 sentence objective shown above the
    minimap. Drops jargon and the structured WIN/LOSE machine block —
    those live behind the **Details** expand instead."""
    if sess is None:
        return ""
    from openra_bench.playlist import simplify_objective

    raw = (getattr(sess, "objective", "") or "").strip()
    if not raw:
        return ""
    plain = simplify_objective(raw)
    if not plain:
        plain = simplify_objective(raw, max_chars=600)
    return f"### 🎯 Your goal\n{_md_escape(plain)}"


def _playlist_details_md(sess) -> str:
    """The full briefing (jargon, win clauses, fail clauses) tucked
    behind the **Details** expand. Same content `_play_briefing_md`
    shows in the Play tab — an interested tester can still drill in."""
    if sess is None:
        return ""
    try:
        from openra_bench.human_labeling import HumanController
        from openra_bench.playlist import simplify_text

        brief = HumanController._briefing(sess.render_state())
    except Exception:  # noqa: BLE001
        brief = ""
    plain_brief = simplify_text(brief)
    obj = (getattr(sess, "objective", "") or "").strip()
    plain_obj = simplify_text(obj)
    return (
        f"**Full objective**\n\n{plain_obj}\n\n"
        f"**Per-turn briefing (model sees the same):**\n```\n"
        f"{plain_brief}\n```"
    )


def _playlist_status_md(sess, st: dict) -> str:
    """Status line under the minimap. After a game ends, append the
    auto-advance countdown so the player knows the next scenario is
    coming.

    On game-over the Playlist tab COLOURS the outcome token (red for
    LOSS, green for WIN, gray for DRAW — `OUTCOME_COLORS`) and APPENDS
    a plain-English explanation sentence ("You ran out of time." /
    "Your last unit was destroyed." / "You reached the objective.")
    so a non-gamer reading the bare ``**LOSS**`` token knows WHY the
    game ended. The chip uses inline HTML; Gradio Markdown renders it
    natively. (B9 Playlist nits #1 + #2.)"""
    if sess is None or not getattr(sess, "done", False):
        return _play_status_md(sess)
    from openra_bench.playlist import (
        AUTO_ADVANCE_WAIT_SECONDS, explain_outcome, outcome_html,
    )

    status = sess.status()
    outcome = str(status.get("outcome", "draw") or "draw").lower()
    own_units = None
    has_base = None
    try:
        rs = sess.render_state() or {}
        # `units_summary` is the CURRENT-frame agent unit list (the
        # accumulating `own_building_types` set on signals would lie
        # about a destroyed base). `own_buildings` is the parallel
        # current-frame agent-building list, with `fact` as the base.
        own_units = len(rs.get("units_summary") or [])
        bld_types = {
            str(b.get("type", "")).lower()
            for b in (rs.get("own_buildings") or [])
        }
        has_base = "fact" in bld_types
    except Exception:  # noqa: BLE001
        # Engine signals unavailable — fall back to the time/generic
        # branches in `explain_outcome`.
        pass
    explanation = explain_outcome(
        outcome,
        turn=int(status.get("turn", 0) or 0),
        max_turns=int(status.get("max_turns", 0) or 0),
        own_units=own_units,
        has_base=has_base,
    )
    chip = outcome_html(outcome)
    # Build the head line by hand so the coloured chip replaces the
    # `**OUTCOME**` markdown the Play tab status line emits — keeping
    # the same `Turn n/m · tick t · CHIP — game over.` shape.
    head = (
        f"**Turn {status['turn']}/{status['max_turns']}** · "
        f"tick {status['tick']} · {chip} — game over. {explanation}"
    )
    if status.get("save_path"):
        head += (
            f"\n\n_Run saved (standard playback format): "
            f"`{status['save_path']}`_"
        )

    pl = st.get("playlist") or []
    idx = int(st.get("idx", 0) or 0)
    n = len(pl)
    if idx + 1 >= n:
        tail = "Final scenario complete — see your **Session summary** below."
    else:
        tail = (
            f"**Game {idx + 1} of {n}: {chip}** — next scenario in "
            f"{int(AUTO_ADVANCE_WAIT_SECONDS)}s. Click **Skip wait ▶** to go now."
        )
    return f"{head}\n\n{tail}"


def _playlist_should_show_build(sess) -> bool:
    """Hide the Build textbox unless the active pack actually exposes
    a build verb — keeps the UI sparse for the move-and-shoot majority.
    """
    if sess is None:
        return False
    try:
        from openra_bench.playlist import needs_build_tool

        compiled = getattr(sess, "compiled", None)
        meta = getattr(compiled, "meta", None) if compiled else None
        # Pack-level `tools:` lives on compile_config / pack.base.
        tools = (
            getattr(compiled, "tools", None)
            or getattr(getattr(compiled, "scenario", None), "tools", None)
            or []
        )
        return needs_build_tool(tools or []) and not getattr(sess, "done", False)
    except Exception:  # noqa: BLE001
        return False


def _playlist_simplified_units_df(sess, sel):
    """Same as `_play_units_df` but with the `type` column run through
    the jargon dictionary so the table shows 'medium tank' instead of
    '2tnk'. Engine-side ids are unchanged — the click-to-select path
    still works."""
    df = _play_units_df(sess, sel)
    if df is None or not len(df) or "type" not in df.columns:
        return df
    try:
        from openra_bench.playlist import simplify_text

        df = df.copy()
        df["type"] = df["type"].map(
            lambda t: simplify_text(str(t)) if t is not None else t
        )
        return df
    except Exception:  # noqa: BLE001
        return df


def _playlist_summary_df(st: dict):
    """Session-end summary table — one row per game played so far. Used
    in the 'Session summary' panel that appears once the playlist is
    complete (or the player clicks 'End session')."""
    rows = list(st.get("results") or [])
    if not rows:
        return pd.DataFrame(columns=[
            "Game", "Scenario", "Level", "Outcome", "Turns", "Max Turns",
        ])
    return pd.DataFrame(rows)


def _playlist_render(st: dict, prev_sess):
    """Open a Playlist session for `st`'s current cell and render it.
    Mirrors `_study_render` but produces the simplified Playlist
    panel set.

    Returns visibility updates for the start / skip-wait / play-row
    buttons — pre-session only `Start` shows; mid-session only the play
    buttons show; on game-over `Skip wait` surfaces until the auto-
    advance timer fires. A non-gamer never sees a button that does
    nothing on click (smoke-test friction fix)."""
    if prev_sess is not None:
        try:
            prev_sess.close()
        except Exception:  # noqa: BLE001
            pass
    empty_units = _play_units_df(None, [])
    pl = st.get("playlist") or []
    idx = int(st.get("idx", 0) or 0)
    # Default visibility — overridden per branch below.
    vis_start = gr.update(visible=True)        # Start playlist
    vis_skip = gr.update(visible=False)        # Skip wait ▶
    vis_play_row = gr.update(visible=False)    # End turn / clear / cancel
    if idx >= len(pl):
        # Playlist complete — return the summary view, no session.
        return (
            None, [], [], "", None, "", "_session complete_",
            empty_units, st, _playlist_progress_md(st),
            _playlist_summary_df(st), gr.update(visible=True),
            gr.update(visible=False),
            vis_start, vis_skip, vis_play_row,
        )
    pack, level = pl[idx]
    try:
        from openra_bench.human_labeling import InteractiveSession

        sess = InteractiveSession.from_pack(
            pack, level, seed=1, player=st.get("player", "anon"),
        )
    except Exception as e:  # noqa: BLE001
        # Engine wheel missing — friendly message, never crash.
        msg = (
            f"⚠️ Could not start `{pack}:{level}` — `{e}`.\n\n"
            "Tip: from the repo root, run "
            "`cd OpenRA-Rust && PATH=$HOME/.cargo/bin:/opt/anaconda3/bin:$PATH "
            "maturin develop --release` first to install the engine wheel."
        )
        return (
            None, [], [], msg, None, "",
            "_engine wheel not installed — see the message above_",
            empty_units, st, _playlist_progress_md(st),
            _playlist_summary_df(st), gr.update(visible=False),
            gr.update(visible=False),
            vis_start, vis_skip, vis_play_row,
        )
    img, _brief, _status, _units = _play_render(sess, [], [], False)
    # In-session: hide Start (re-clicking it would silently re-launch
    # from scratch and lose progress); show End-turn / clear row;
    # surface Skip-wait only once the game is over.
    vis_start = gr.update(visible=False)
    vis_play_row = gr.update(visible=True)
    vis_skip = gr.update(visible=bool(getattr(sess, "done", False)))
    return (
        sess, [], [], _playlist_objective_md(sess), img,
        _playlist_details_md(sess), _playlist_status_md(sess, st),
        _playlist_simplified_units_df(sess, []), st,
        _playlist_progress_md(st), _playlist_summary_df(st),
        gr.update(visible=False),  # summary panel hidden during play
        gr.update(visible=_playlist_should_show_build(sess)),
        vis_start, vis_skip, vis_play_row,
    )


def _playlist_start(prev_sess, player):
    """The single 'Start playlist' click — set up state, open game 1."""
    st = _playlist_state(player)
    return _playlist_render(st, prev_sess)


def _playlist_record_outcome(sess, st: dict) -> dict:
    """If the current session has finished and we haven't recorded
    its result yet, append it to `st['results']`. Returns the updated
    state dict (may be the same object)."""
    if sess is None:
        return st
    if not getattr(sess, "done", False):
        return st
    pl = st.get("playlist") or []
    idx = int(st.get("idx", 0) or 0)
    results = list(st.get("results") or [])
    if idx >= len(pl) or idx < len(results):
        return st  # already recorded for this idx
    pack, level = pl[idx]
    from openra_bench.playlist import session_summary_row

    results.append(session_summary_row(
        idx=idx, pack=pack, level=level,
        outcome=str(getattr(sess, "outcome", "draw") or "draw"),
        turns=int(getattr(sess, "turn", 0) or 0),
        max_turns=int(getattr(sess, "max_turns", 0) or 0),
    ))
    new_st = dict(st)
    new_st["results"] = results
    if new_st.get("done_at") is None:
        new_st["done_at"] = time.time()
    return new_st


def _playlist_click(sess, sel, queue, show_obj, evt: gr.SelectData):
    """Wrapper around `_play_click` that swaps the unit-table output
    for the jargon-simplified version. Matches `_play_click`'s 5-output
    signature so it drops in to the same `pl_img.select` wiring.

    The `evt: gr.SelectData` annotation is REQUIRED — without it Gradio
    treats `evt` as an explicit input and the wiring (which only passes
    4 explicit inputs) silently mismatches, dropping the click event so
    the minimap becomes inert. Pinned by the Playlist smoke-test."""
    new_sel, new_queue, brief, _units, img = _play_click(
        sess, sel, queue, show_obj, evt
    )
    return (
        new_sel, new_queue, brief,
        _playlist_simplified_units_df(sess, new_sel), img,
    )


def _playlist_clear_selection(sess, queue, show_obj):
    sel, brief, _units, img = _play_clear_selection(sess, queue, show_obj)
    return sel, brief, _playlist_simplified_units_df(sess, sel), img


def _playlist_clear_queue(sess, sel, show_obj):
    queue, brief, _units, img = _play_clear_queue(sess, sel, show_obj)
    return queue, brief, _playlist_simplified_units_df(sess, sel), img


def _playlist_build_item(sess, sel, queue, item, show_obj):
    queue, item_out, brief, _units, img = _play_build_item(
        sess, sel, queue, item, show_obj
    )
    return (
        queue, item_out, brief,
        _playlist_simplified_units_df(sess, sel), img,
    )


def _playlist_end_turn(sess, sel, queue, st):
    """End-turn for Playlist mode. After advancing the engine, if the
    game just ended, stamp `done_at` so the auto-advance countdown can
    fire AND surface the Skip-wait button so the tester can advance
    sooner than the 5s timer (it's hidden mid-game)."""
    if sess is not None and not sess.done:
        try:
            sess.submit_turn(list(queue))
        except Exception as e:  # noqa: BLE001
            logger.warning("playlist submit_turn failed: %s", e)
    img, _brief, _status, _units = _play_render(sess, [], [], False)
    new_st = _playlist_record_outcome(sess, st or {})
    skip_visible = bool(getattr(sess, "done", False))
    return (
        sess, [], [], img, _playlist_details_md(sess),
        _playlist_status_md(sess, new_st),
        _playlist_simplified_units_df(sess, []), new_st,
        gr.update(visible=_playlist_should_show_build(sess)),
        gr.update(visible=skip_visible),
    )


def _playlist_advance(prev_sess, st):
    """Move to the next pack — fired by the auto-advance Timer or the
    manual 'Skip wait ▶' button."""
    if not st or "playlist" not in st:
        return _playlist_render(_playlist_state(), prev_sess)
    new_st = dict(st)
    new_st["idx"] = int(new_st.get("idx", 0) or 0) + 1
    new_st["done_at"] = None
    return _playlist_render(new_st, prev_sess)


_PL_TICK_NOOP_OUTPUTS = 16  # length of _pl_render_outs (incl. button-vis trio)


def _playlist_tick(sess, st):
    """Auto-advance Timer tick. If the current game has been over for
    >= AUTO_ADVANCE_WAIT_SECONDS seconds, advance to the next pack;
    otherwise return state unchanged. The Timer fires every 1s while
    the tab is active."""
    from openra_bench.playlist import (
        AUTO_ADVANCE_WAIT_SECONDS,
        playlist_should_advance,
    )

    noop = tuple(gr.skip() for _ in range(_PL_TICK_NOOP_OUTPUTS))
    if not st or not isinstance(st, dict):
        return noop
    if sess is None or not getattr(sess, "done", False):
        return noop
    done_at = st.get("done_at")
    if not playlist_should_advance(
        True, done_at, time.time(), AUTO_ADVANCE_WAIT_SECONDS
    ):
        return noop
    return _playlist_advance(sess, st)


def _playlist_submit_baseline(st):
    """Post the playlist's per-game outcomes to the bench raw-games log
    as `agent_type=Human`. Idempotent — a second click is a no-op."""
    if not st:
        return "_no session to submit_", st
    if st.get("submitted"):
        return "_already submitted_", st
    rows = list(st.get("results") or [])
    if not rows:
        return "_no completed games yet — play at least one scenario_", st
    player = (st.get("player") or "anon").strip() or "anon"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    saved = 0
    for r in rows:
        outcome = str(r.get("Outcome", "DRAW")).lower()
        # The Submit tab consumes a richer JSON blob than we have here;
        # mirror its expected shape so the leaderboard pipeline picks
        # the rows up identically to a model run.
        try:
            _save_raw_game({
                "agent_name": player,
                "agent_type": "Human",
                "opponent": "Beginner",
                "scenario": f"{r.get('Scenario')}:{r.get('Level')}",
                "result": outcome,
                "win": outcome == "win",
                "kills_cost": 0,
                "deaths_cost": 0,
                "assets_value": 0,
                "ticks": int(r.get("Turns", 0) or 0) * 90,
                "timestamp": timestamp,
                "replay_url": "",
                "agent_url": "",
                "hf_username": "",
            })
            saved += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("playlist submit row failed: %s", e)
    new_st = dict(st)
    new_st["submitted"] = True
    return (
        f"✅ Submitted {saved} game(s) for `{player}` as **Human** baseline. "
        f"They will appear in the leaderboard once aggregated.",
        new_st,
    )


def build_app() -> gr.Blocks:
    """Build the Gradio leaderboard app."""
    initial_df = add_type_badges(load_data())

    with gr.Blocks(title="OpenRA-Bench") as app:
        gr.Markdown(
            "# OpenRA-Bench\n"
            "**Agent Leaderboard for OpenRA-RL** — "
            "Train AI to Play Real-Time Strategy"
        )

        with gr.Tabs():
            # ── Leaderboard Tab ───────────────────────────────────────────
            with gr.Tab("Leaderboard"):
                with gr.Row():
                    search_box = gr.Textbox(
                        label="Search agents",
                        placeholder="Search by name (supports regex, comma-separated)...",
                        scale=3,
                    )
                    type_filter = gr.CheckboxGroup(
                        choices=["Scripted", "LLM", "RL"],
                        value=["Scripted", "LLM", "RL"],
                        label="Agent Type",
                        scale=2,
                    )
                    opponent_filter = gr.Dropdown(
                        choices=["All", "Beginner", "Easy", "Medium", "Normal", "Hard"],
                        value="All",
                        label="Opponent",
                        scale=1,
                    )
                    show_unverified = gr.Checkbox(
                        label="Show unverified",
                        value=True,
                        scale=1,
                    )

                leaderboard = gr.Dataframe(
                    value=initial_df,
                    datatype=[
                        "number",    # Rank
                        "html",      # Agent (may contain hyperlink)
                        "html",      # Type (badge)
                        "html",      # Status (verified badge)
                        "str",       # Opponent
                        "number",    # Games
                        "number",    # Win Rate
                        "number",    # Score
                        "number",    # K/D Ratio
                        "number",    # Avg Kills
                        "number",    # Avg Deaths
                        "number",    # Avg Economy
                        "number",    # Avg Game Length
                        "str",       # Date
                        "html",      # Replay (download link)
                    ],
                    interactive=False,
                    show_label=False,
                )

                # Wire up filters
                filter_inputs = [search_box, type_filter, opponent_filter, show_unverified]
                for component in filter_inputs:
                    component.change(
                        fn=filter_leaderboard,
                        inputs=filter_inputs,
                        outputs=leaderboard,
                    )

            # ── Capability Leaderboard Tab ────────────────────────────────
            # run_eval reports (composite + Perception/Reasoning/Action +
            # weakest link) published via `run_eval --leaderboard`.
            with gr.Tab("Capability Leaderboard"):
                gr.Markdown(
                    "Models on customized scenarios, scored on the "
                    "Perception→Reasoning→Action chain. **weakest_link** "
                    "shows the dominant failure mode."
                )
                cap_df = gr.Dataframe(
                    value=load_capability_leaderboard(),
                    interactive=False,
                    wrap=True,
                )
                refresh_cap = gr.Button("Refresh")
                refresh_cap.click(load_capability_leaderboard, outputs=cap_df)

            # ── Scenarios Tab ─────────────────────────────────────────────
            with gr.Tab("Scenarios"):
                gr.Markdown(
                    "Browse every active scenario pack. Each pack tests "
                    "one capability at three difficulty levels with "
                    "identical win/fail rules used to score LLM agents. "
                    "Switch to **中文** for Chinese translations."
                )
                with gr.Row():
                    scen_search = gr.Textbox(
                        label="Search",
                        placeholder="Filter by id, title, or meaning...",
                        scale=3,
                    )
                    scen_cap_filter = gr.CheckboxGroup(
                        choices=["perception", "reasoning", "action",
                                 "adversarial"],
                        value=["perception", "reasoning", "action",
                               "adversarial"],
                        label="Capability",
                        scale=3,
                    )
                    scen_lang = gr.Radio(
                        choices=["English", "中文"],
                        value="English",
                        label="Language",
                        scale=1,
                    )
                scen_table = gr.Dataframe(
                    value=_scenarios_filter("", [
                        "perception", "reasoning", "action", "adversarial"
                    ]),
                    interactive=False,
                    wrap=True,
                    show_label=False,
                )
                scen_filter_inputs = [scen_search, scen_cap_filter]
                for comp in scen_filter_inputs:
                    comp.change(
                        fn=_scenarios_filter,
                        inputs=scen_filter_inputs,
                        outputs=scen_table,
                    )
                gr.Markdown("---")
                scen_id_input = gr.Textbox(
                    label="Pack ID (click a row above or type)",
                    placeholder="e.g. combat-focus-fire-priority",
                )
                scen_detail = gr.Markdown(
                    "_Select a scenario from the table above to see "
                    "details._"
                )

                def _scen_detail_with_lang(pack_id, lang_choice):
                    lang = "zh" if lang_choice == "中文" else "en"
                    return _scenarios_detail_md(pack_id, lang)

                scen_id_input.change(
                    fn=_scen_detail_with_lang,
                    inputs=[scen_id_input, scen_lang],
                    outputs=scen_detail,
                )
                scen_lang.change(
                    fn=_scen_detail_with_lang,
                    inputs=[scen_id_input, scen_lang],
                    outputs=scen_detail,
                )

                def _scen_row_select(evt: gr.SelectData, df, lang_choice):
                    if evt is None or df is None or not len(df):
                        return gr.update(), gr.update()
                    try:
                        row_idx = evt.index[0]
                        pack_id = str(df.iloc[row_idx]["ID"])
                        lang = "zh" if lang_choice == "中文" else "en"
                        return pack_id, _scenarios_detail_md(pack_id, lang)
                    except Exception:  # noqa: BLE001
                        return gr.update(), gr.update()

                scen_table.select(
                    fn=_scen_row_select,
                    inputs=[scen_table, scen_lang],
                    outputs=[scen_id_input, scen_detail],
                )

            # ── Battle Viewer Tab ─────────────────────────────────────────
            # Browse saved playbacks: filter run → model → scenario,
            # step the battle with ◀ / ▶, and compare two models
            # head-to-head on the same scenario+seed.
            with gr.Tab("Battle Viewer"):
                gr.Markdown(
                    "Pick a **run → model → scenario**, then step the "
                    f"battle. Playback root: `{PLAYBACK_ROOT}` "
                    "(set `OPENRA_BENCH_PLAYBACK_ROOT` to change)."
                )
                bv_idx = gr.State([])
                bv_turn = gr.State(0)
                with gr.Row():
                    bv_run = gr.Dropdown(label="Run", scale=2)
                    bv_model = gr.Dropdown(label="Model", scale=2)
                    bv_scen = gr.Dropdown(label="Scenario @ seed", scale=3)
                    bv_refresh = gr.Button("⟳ Rescan", scale=1)
                with gr.Row():
                    bv_compare = gr.Checkbox(label="Compare mode", value=False)
                    bv_bsel = gr.Dropdown(
                        label="B: run / model (same scenario+seed)", scale=3
                    )
                with gr.Row():
                    bv_prev = gr.Button("◀ Prev turn")
                    bv_pos = gr.Markdown("—")
                    bv_next = gr.Button("Next turn ▶")
                with gr.Row():
                    with gr.Column():
                        bv_a_img = gr.Image(
                            label="A minimap", height=320,
                            show_label=True, interactive=False
                        )
                        bv_a_md = gr.Markdown()
                    with gr.Column():
                        bv_b_img = gr.Image(
                            label="B minimap", height=320,
                            show_label=True, interactive=False
                        )
                        bv_b_md = gr.Markdown()

                _render_outs = [
                    bv_a_img, bv_a_md, bv_b_img, bv_b_md, bv_pos, bv_bsel
                ]
                _sel = [bv_run, bv_model, bv_scen]

                def _bv_go(idx, run, model, scen, turn, comp, b, delta=0):
                    turn = max(0, (turn or 0) + delta)
                    *outs, bupd = _bv_render(
                        idx, run, model, scen, turn, comp, b
                    )
                    return (*outs, bupd, turn)

                bv_refresh.click(
                    bv_runs, outputs=[bv_idx, bv_run]
                ).then(
                    bv_on_run, [bv_idx, bv_run], bv_model
                ).then(
                    bv_on_model, [bv_idx, bv_run, bv_model], bv_scen
                ).then(
                    _bv_go,
                    [bv_idx, bv_run, bv_model, bv_scen, bv_turn,
                     bv_compare, bv_bsel],
                    [*_render_outs, bv_turn],
                )

                bv_run.change(bv_on_run, [bv_idx, bv_run], bv_model).then(
                    bv_on_model, [bv_idx, bv_run, bv_model], bv_scen
                )
                bv_model.change(
                    bv_on_model, [bv_idx, bv_run, bv_model], bv_scen
                )
                for comp in (bv_scen, bv_compare, bv_bsel):
                    comp.change(
                        lambda i, r, m, s, c, b: _bv_go(
                            i, r, m, s, 0, c, b),
                        [bv_idx, bv_run, bv_model, bv_scen, bv_compare,
                         bv_bsel],
                        [*_render_outs, bv_turn],
                    )
                bv_prev.click(
                    lambda i, r, m, s, t, c, b: _bv_go(
                        i, r, m, s, t, c, b, -1),
                    [bv_idx, bv_run, bv_model, bv_scen, bv_turn,
                     bv_compare, bv_bsel],
                    [*_render_outs, bv_turn],
                )
                bv_next.click(
                    lambda i, r, m, s, t, c, b: _bv_go(
                        i, r, m, s, t, c, b, +1),
                    [bv_idx, bv_run, bv_model, bv_scen, bv_turn,
                     bv_compare, bv_bsel],
                    [*_render_outs, bv_turn],
                )
                app.load(bv_runs, outputs=[bv_idx, bv_run]).then(
                    bv_on_run, [bv_idx, bv_run], bv_model
                ).then(
                    bv_on_model, [bv_idx, bv_run, bv_model], bv_scen
                )

            # ── Play Tab (human-labeling machine) ─────────────────────────
            # Play the exact scenarios LLM agents are scored on, by
            # clicking the minimap. Backed by InteractiveSession so a
            # human's run is graded by the identical win/fail rules —
            # human-vs-LLM comparison on one bench.
            with gr.Tab("Play"):
                gr.Markdown(
                    "Play a scenario yourself — the same scenarios LLM "
                    "agents are scored on. Pick a **scenario → level → "
                    "seed**, click **Start**. Then, on the minimap: "
                    "**click your own unit** to select it (click again "
                    "to deselect, click several to build a group); with "
                    "units selected, **click empty ground to move** "
                    "them there, or **click an enemy to attack** it. "
                    "**End Turn** advances. You are graded by the "
                    "identical win/fail rules as the models."
                )
                play_sess = gr.State(None)
                play_sel = gr.State([])
                play_queue = gr.State([])
                study_state = gr.State({})
                with gr.Accordion(
                    "📋 Human-study mode — 24-pack subset, 3 conditions",
                    open=False,
                ):
                    gr.Markdown(
                        "For the **human-baseline study**. Enter your name "
                        "and click **Begin study** — you'll be walked "
                        "through 72 games (24 scenarios × fog / no-fog / "
                        "handoff-deficit), counterbalanced per player. "
                        "Play each to game-over, then **Next scenario ▶**. "
                        "Every game auto-saves apples-to-apple with the "
                        "model runs."
                    )
                    with gr.Row():
                        study_player = gr.Textbox(
                            label="Your name / id", scale=2,
                        )
                        study_begin_btn = gr.Button("Begin study", scale=1)
                        study_next_btn = gr.Button(
                            "Next scenario ▶", variant="primary", scale=1,
                        )
                    study_progress = gr.Markdown()
                with gr.Row():
                    play_scen = gr.Dropdown(
                        choices=_play_scenarios(), label="Scenario",
                        scale=3,
                    )
                    play_level = gr.Dropdown(
                        choices=_PLAY_LEVELS, value="easy",
                        label="Level", scale=1,
                    )
                    play_seed = gr.Number(
                        value=1, label="Seed", precision=0, scale=1,
                    )
                    play_start = gr.Button("▶ Start", scale=1)
                play_show_objectives = gr.Checkbox(
                    label="Show objective rings", value=False,
                    info="Only available when the scenario already reveals exact coordinates.",
                )
                gr.Markdown(
                    "_Press **Enter** to end the turn when focus is outside "
                    "inputs._"
                )
                play_objective = gr.Markdown()
                play_status = gr.Markdown(_play_status_md(None))
                # Minimap on its own full-width row. No fixed height —
                # it renders at its natural aspect so there is no
                # letterbox band wasting space.
                play_img = gr.Image(
                    label="Minimap — click your unit, then click where "
                    "to send it",
                    interactive=False, show_label=True,
                )
                play_brief = gr.Markdown()
                play_units = gr.Dataframe(
                    label="Your units (▶ = selected, shown at top)",
                    headers=_PLAY_UNIT_COLS,
                    interactive=False, wrap=True,
                )
                with gr.Row():
                    play_build_item = gr.Textbox(
                        label="Build item",
                        value="e1",
                        placeholder="e1, pbox, proc, powr, ...",
                        scale=2,
                    )
                    play_build_btn = gr.Button("Queue build", scale=1)
                with gr.Row():
                    play_clearsel_btn = gr.Button(
                        "✖ Clear selected units", scale=1
                    )
                    play_clear_btn = gr.Button(
                        "Cancel queued orders", scale=1
                    )
                    play_end_btn = gr.Button(
                        "End Turn (Enter) ▶", variant="primary", scale=2,
                        elem_id="play-end-turn-btn",
                    )

                play_start.click(
                    _play_start,
                    inputs=[
                        play_sess, play_scen, play_level, play_seed,
                        play_show_objectives,
                    ],
                    outputs=[
                        play_sess, play_sel, play_queue, play_objective,
                        play_img, play_brief, play_status, play_units,
                    ],
                )
                play_img.select(
                    _play_click,
                    inputs=[
                        play_sess, play_sel, play_queue,
                        play_show_objectives,
                    ],
                    outputs=[
                        play_sel, play_queue, play_brief, play_units,
                        play_img,
                    ],
                )
                play_build_btn.click(
                    _play_build_item,
                    inputs=[
                        play_sess, play_sel, play_queue, play_build_item,
                        play_show_objectives,
                    ],
                    outputs=[
                        play_queue, play_build_item, play_brief,
                        play_units, play_img,
                    ],
                )
                _study_outputs = [
                    play_sess, play_sel, play_queue, play_objective,
                    play_img, play_brief, play_status, play_units,
                    study_state, study_progress,
                ]
                study_begin_btn.click(
                    _study_begin, inputs=[play_sess, study_player],
                    outputs=_study_outputs,
                )
                study_next_btn.click(
                    _study_next, inputs=[play_sess, study_state],
                    outputs=_study_outputs,
                )
                play_end_btn.click(
                    _play_end_turn,
                    inputs=[
                        play_sess, play_sel, play_queue,
                        play_show_objectives,
                    ],
                    outputs=[
                        play_sess, play_sel, play_queue,
                        play_img, play_brief, play_status, play_units,
                    ],
                )
                play_clearsel_btn.click(
                    _play_clear_selection,
                    inputs=[play_sess, play_queue, play_show_objectives],
                    outputs=[
                        play_sel, play_brief, play_units, play_img,
                    ],
                )
                play_clear_btn.click(
                    _play_clear_queue,
                    inputs=[play_sess, play_sel, play_show_objectives],
                    outputs=[
                        play_queue, play_brief, play_units, play_img,
                    ],
                )
                play_show_objectives.change(
                    _play_toggle_objectives,
                    inputs=[
                        play_sess, play_sel, play_queue,
                        play_show_objectives,
                    ],
                    outputs=play_img,
                )
                app.load(fn=None, js=_PLAY_KEYBOARD_JS)

            # ── Playlist Tab (cold-start non-gamer UX) ───────────────────
            # The Play tab is a sandbox / power-user UI: 210 packs in a
            # dropdown, full briefings, full tool palette. The Playlist
            # tab is the COLD-START non-gamer UX: one click to start, a
            # curated 20-pack list (no dropdown — the player never picks
            # a pack), a 1-2 sentence plain-English objective above the
            # minimap (no jargon), only move/attack/build/end-turn, an
            # auto-advance countdown when a game ends, and a session
            # summary with a 'Submit baseline' button at the end.
            #
            # Underneath this is the same `InteractiveSession` the Play
            # tab uses (and the 24-pack study uses), so the playback /
            # leaderboard format is apples-to-apple with model runs on
            # the same scenarios.
            with gr.Tab("Playlist"):
                gr.Markdown(
                    "## Beginner Playlist — 20 scenarios, ~60-90 minutes\n\n"
                    "A curated tour of the bench, hand-picked for "
                    "non-gamers: visible green tanks, visible red "
                    "enemies, drive units to yellow rings or attack "
                    "what's in front of you. **No game knowledge "
                    "needed.** Enter your name, hit **Start**, and play "
                    "each scenario to game-over — the next one loads "
                    "automatically.\n\n"
                    "**How to play.** Click your green unit to select "
                    "it (click again to deselect; click several to "
                    "build a group). With units selected, click empty "
                    "ground to **move** there, or click a red enemy "
                    "to **attack** it. **End Turn** advances time. The "
                    "minimap shows yellow rings where you need to go."
                )
                pl_sess = gr.State(None)
                pl_sel = gr.State([])
                pl_queue = gr.State([])
                pl_state = gr.State({})
                # A constant 'show objectives = False' state — reused by
                # `_play_click` / `_play_clear_*` so the Playlist tab can
                # share Play-tab handlers without the objective-rings
                # toggle (objectives are surfaced via the plain-English
                # text above the minimap, not as on-map rings).
                pl_show_obj_const = gr.State(False)

                pl_progress = gr.Markdown(_playlist_progress_md({}))
                with gr.Row():
                    pl_player = gr.Textbox(
                        label="Your name (required)",
                        placeholder="e.g. Alex",
                        scale=3,
                    )
                    # Start is the only button visible pre-session;
                    # `_playlist_render` hides it once a game is open.
                    # Disabled until the name textbox holds ≥1 non-
                    # whitespace character — empty input previously
                    # silently became "anon" in the recorded summary
                    # row, which a non-gamer never opted into. (B9
                    # Playlist nit #3.)
                    pl_start_btn = gr.Button(
                        "▶ Start playlist", variant="primary", scale=1,
                        visible=True, interactive=False,
                    )
                    # Skip-wait is hidden until a game ends — clicking
                    # it before that did nothing useful and confused a
                    # smoke-tester ("which one starts? Start or Skip?").
                    # `_playlist_render` / `_playlist_end_turn` toggle
                    # it visible on game-over.
                    pl_skip_btn = gr.Button(
                        "Skip wait ▶", scale=1, visible=False,
                    )

                pl_objective = gr.Markdown()
                pl_status = gr.Markdown(_play_status_md(None))
                pl_img = gr.Image(
                    label="Minimap — click your green unit, then click "
                    "where to send it (move) or click a red enemy (attack)",
                    interactive=False, show_label=True,
                )
                pl_units = gr.Dataframe(
                    label="Your units (▶ = selected, shown at top)",
                    headers=_PLAY_UNIT_COLS,
                    interactive=False, wrap=True,
                )
                with gr.Accordion("Details (advanced)", open=False):
                    pl_details = gr.Markdown()
                with gr.Row(visible=False) as pl_build_row:
                    pl_build_item = gr.Textbox(
                        label="Build item",
                        placeholder="e.g. e1, pbox",
                        scale=2,
                    )
                    pl_build_btn = gr.Button("Queue build", scale=1)
                # The play-action row is hidden until a session is
                # live — clicking End Turn / Clear / Cancel before
                # Start did nothing useful and tester reports show a
                # non-gamer reads them as "what do I do first" UI noise.
                # `_playlist_render` toggles this row visible once a
                # session opens.
                with gr.Row(visible=False) as pl_play_row:
                    pl_clearsel_btn = gr.Button(
                        "✖ Clear selected units", scale=1,
                    )
                    pl_clear_btn = gr.Button(
                        "Cancel queued orders", scale=1,
                    )
                    pl_end_btn = gr.Button(
                        "End Turn ▶", variant="primary", scale=2,
                    )

                # Session-end summary panel — hidden until the playlist
                # finishes (or `_playlist_render` toggles it).
                with gr.Group(visible=False) as pl_summary_panel:
                    gr.Markdown("### 📊 Session summary")
                    pl_summary_df = gr.Dataframe(
                        headers=["Game", "Scenario", "Level", "Outcome",
                                 "Turns", "Max Turns"],
                        interactive=False, wrap=True,
                    )
                    pl_submit_btn = gr.Button(
                        "Submit baseline", variant="primary",
                    )
                    pl_submit_msg = gr.Markdown()

                # Background timer — fires every 1s while the tab is
                # active. When the current game is over and the
                # 5-second wait has elapsed, advance to the next pack.
                pl_timer = gr.Timer(1.0, active=True)

                _pl_render_outs = [
                    pl_sess, pl_sel, pl_queue, pl_objective, pl_img,
                    pl_details, pl_status, pl_units, pl_state,
                    pl_progress, pl_summary_df, pl_summary_panel,
                    pl_build_row,
                    # Button visibility — smoke-test friction fixes.
                    pl_start_btn, pl_skip_btn, pl_play_row,
                ]
                # B9 nit #3: gate the Start button on a non-empty name.
                # The name flows directly into the InteractiveSession's
                # `player` field (and the recorded raw-game row) so an
                # empty textbox would silently become "anon" — which a
                # non-gamer never opted into. Re-enable as soon as the
                # textbox holds ≥1 non-whitespace character.
                def _pl_validate_name(name):
                    from openra_bench.playlist import is_valid_player_name
                    return gr.update(interactive=is_valid_player_name(name))

                pl_player.change(
                    _pl_validate_name, inputs=[pl_player],
                    outputs=[pl_start_btn],
                )
                pl_start_btn.click(
                    _playlist_start, inputs=[pl_sess, pl_player],
                    outputs=_pl_render_outs,
                )
                pl_skip_btn.click(
                    _playlist_advance, inputs=[pl_sess, pl_state],
                    outputs=_pl_render_outs,
                )
                pl_timer.tick(
                    _playlist_tick, inputs=[pl_sess, pl_state],
                    outputs=_pl_render_outs,
                )

                # Click on minimap → reuse the Play-tab handler (same
                # selection / attack / move semantics). The Playlist tab
                # never shows objective rings (objectives are surfaced
                # via the plain-English text above the map).
                pl_img.select(
                    _playlist_click,
                    inputs=[pl_sess, pl_sel, pl_queue, pl_show_obj_const],
                    outputs=[pl_sel, pl_queue, pl_details, pl_units, pl_img],
                )
                pl_end_btn.click(
                    _playlist_end_turn,
                    inputs=[pl_sess, pl_sel, pl_queue, pl_state],
                    outputs=[
                        pl_sess, pl_sel, pl_queue, pl_img, pl_details,
                        pl_status, pl_units, pl_state, pl_build_row,
                        pl_skip_btn,
                    ],
                )
                pl_clearsel_btn.click(
                    _playlist_clear_selection,
                    inputs=[pl_sess, pl_queue, pl_show_obj_const],
                    outputs=[pl_sel, pl_details, pl_units, pl_img],
                )
                pl_clear_btn.click(
                    _playlist_clear_queue,
                    inputs=[pl_sess, pl_sel, pl_show_obj_const],
                    outputs=[pl_queue, pl_details, pl_units, pl_img],
                )
                pl_build_btn.click(
                    _playlist_build_item,
                    inputs=[
                        pl_sess, pl_sel, pl_queue, pl_build_item,
                        pl_show_obj_const,
                    ],
                    outputs=[
                        pl_queue, pl_build_item, pl_details, pl_units,
                        pl_img,
                    ],
                )
                pl_submit_btn.click(
                    _playlist_submit_baseline, inputs=[pl_state],
                    outputs=[pl_submit_msg, pl_state],
                )

            # ── About Tab ─────────────────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD)

            # ── Submit Tab ────────────────────────────────────────────────
            with gr.Tab("Submit"):
                gr.Markdown(
                    "## Upload Results\n\n"
                    "Upload a bench export JSON from your OpenRA-RL game. "
                    "After each game, the agent saves a JSON file to "
                    "`~/.openra-rl/bench-exports/`."
                )
                with gr.Row():
                    json_upload = gr.File(
                        label="Bench export JSON",
                        file_types=[".json"],
                        scale=3,
                    )
                    replay_upload = gr.File(
                        label="Replay file (optional)",
                        file_types=[".orarep"],
                        scale=2,
                    )
                submit_btn = gr.Button("Submit Results", variant="primary")
                submit_output = gr.Markdown()

                submit_btn.click(
                    fn=handle_upload,
                    inputs=[json_upload, replay_upload],
                    outputs=[submit_output, leaderboard],
                )

                # API endpoint for CLI auto-upload (JSON only)
                api_json_input = gr.Textbox(visible=False)
                api_result = gr.Textbox(visible=False)
                api_btn = gr.Button(visible=False)
                api_btn.click(
                    fn=handle_api_submit,
                    inputs=[api_json_input],
                    outputs=[api_result],
                    api_name="submit",
                )

                # API endpoint for CLI upload with replay
                api_json_input2 = gr.Textbox(visible=False)
                api_replay_input = gr.File(visible=False)
                api_result2 = gr.Textbox(visible=False)
                api_btn2 = gr.Button(visible=False)
                api_btn2.click(
                    fn=handle_api_submit_with_replay,
                    inputs=[api_json_input2, api_replay_input],
                    outputs=[api_result2],
                    api_name="submit_with_replay",
                )

                gr.Markdown(SUBMIT_MD)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        allowed_paths=[str(SUBMISSIONS_DIR), str(PLAYBACK_ROOT)],
    )
