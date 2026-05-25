"""OpenRA-Bench: Agent Leaderboard for OpenRA-RL.

A Gradio app that displays agent rankings, supports filtering by type
and opponent difficulty, and lets users run evaluations in-browser.

Run locally:
    python app.py

Deploy on HuggingFace Spaces:
    Push app.py, requirements.txt, data/, and README.md to your HF Space.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
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
        # `objective` mean-over-runs column dropped: it averaged the
        # per-episode blocking-ratio across an entire run, producing
        # a number with no useful interpretation. Headline metrics
        # are now win_rate + composite; per-episode `leaves_final`
        # is the only honest detail surface (see leaderboard.py).
        "adversarial_rating", "perception", "reasoning",
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
        # Per-leaf table: explicit current/target + satisfied mark.
        # The legacy scalar "objective: 79%" averaged unrelated leaves
        # (e.g. a 0.57 kills leaf + 0.0 deadline-violated leaf → 0.28
        # "near win" when both clauses had failed) — it is dropped
        # in favour of this row-by-row display.
        parts = []
        for leaf in g.get("leaves", []):
            mark = "✅" if leaf.get("satisfied") else "❌"
            cur = leaf.get("current")
            tgt = leaf.get("target")
            name = leaf.get("name", "?")
            if name in ("within_ticks", "after_ticks") and cur is not None:
                parts.append(f"{name} tick {cur}/{tgt} {mark}")
            else:
                parts.append(f"{name} {cur}/{tgt} {mark}")
        bars = " · ".join(parts)
        rv = g.get("reward_vector", {})
        lines += [
            "**objective leaves:**" + ("  ✅ WON" if g.get("won") else ""),
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

**Minimum games**: 5 games required per agent+opponent to appear on the leaderboard.

### Agent Types

- **Scripted**: Rule-based bots with hardcoded strategies
- **LLM**: Language model agents (Claude, GPT, etc.)
- **RL**: Reinforcement learning policies (PPO, SAC, etc.)

### Links

- [OpenRA-RL Documentation](https://openra-rl.dev)
- [GitHub Repository](https://github.com/yxc20089/OpenRA-RL)
- [OpenRA-Bench Source](https://github.com/yxc20089/OpenRA-Bench)
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


def _md_escape(text: str) -> str:
    """Escape Markdown-significant characters so scenario prose renders
    literally — e.g. the `~` in 'NE ~110,6' must not become strikethrough."""
    text = text.replace("\\", "\\\\")
    for ch in ("~", "*", "_", "`", "#"):
        text = text.replace(ch, "\\" + ch)
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

            # ── About Tab ─────────────────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD)


    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        allowed_paths=[str(PLAYBACK_ROOT)],
    )
