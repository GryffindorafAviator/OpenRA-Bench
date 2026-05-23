"""Playback viewer (pipeline step 7, read side).

`load_episode` is the pure data layer — it reassembles a saved episode
(manifest + turns + transcript + per-turn minimap PNG paths) into one
dict, and is unit-tested. `render_streamlit` is a thin optional UI on
top of it (mirrors the training repo's Streamlit pipeline viewer):

    pip install streamlit
    streamlit run scripts/view_playback.py -- <playback_root>

Per turn it shows: the minimap, the user briefing, the model's
reasoning + tool calls, the signal snapshot, and the goal tracker
(win-condition leaf bars + the cumulative reward vector).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_episode(ep_dir: str | Path) -> dict:
    """Reassemble one ``seed<N>`` episode folder. Tolerant of a still-
    running episode (missing files become empty).

    Accepts EITHER a legacy seed dir OR a FullPlayback audit JSONL file;
    a path ending in `.jsonl` dispatches to `load_audit_jsonl`."""
    p = Path(ep_dir)
    if p.is_file() and p.suffix == ".jsonl":
        return load_audit_jsonl(p)
    d = p
    manifest = _read_json(d / "manifest.json", {})
    messages = _read_json(d / "messages.json", [])
    turns = []
    tj = d / "turns.jsonl"
    if tj.exists():
        for line in tj.read_text().splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                png = d / f"minimap_turn{rec.get('turn', 0):03d}.png"
                rec["minimap_png"] = str(png) if png.exists() else None
                turns.append(rec)
    return {"dir": str(d), "manifest": manifest, "turns": turns,
            "messages": messages}


def find_episodes(root: str | Path) -> list[Path]:
    """All episode targets under a playback root.

    Returns a mix of two shapes the viewer transparently handles:
      * Legacy `.../seed<N>/` dirs (manifest.json + turns.jsonl + …)
      * Audit-format `.../<stem>.jsonl` files written by FullPlayback —
        the loader detects a `.jsonl` path and translates it on the fly.
    """
    root = Path(root)
    legacy = sorted(
        p.parent for p in root.glob("**/manifest.json")
    ) or sorted(root.glob("**/seed*"))
    # Audit-format cells: `<root>/<run_dir>/<pack>__<level>__seedN__fog.jsonl`.
    # Skip the `.partial` half-runs and any sidecar files (start with `_`).
    audit = sorted(
        p for p in root.glob("**/*.jsonl")
        if not p.name.startswith("_")
        and "__seed" in p.name
        and not p.name.endswith(".partial")
    )
    return legacy + audit


def load_audit_jsonl(jsonl_path: str | Path) -> dict:
    """Translate a FullPlayback audit JSONL into the same `{dir, manifest,
    turns, messages}` dict shape `load_episode` returns, so the same
    viewer code (`render_streamlit` / downstream loaders) can consume
    either format transparently.

    - `manifest` is reconstructed from the terminal record's
      `terminal.manifest` block (+ a few top-level fields), preserving
      compatibility with viewers that expect `outcome`, `model`, etc.
    - `turns` mirror the legacy per-turn shape (`turn`, `tick`,
      `commands`, `signals`, `goal`, `minimap_png` path).
    - `messages` is synthesized from the system_prompt + each turn's
      briefing/response, so the viewer's transcript pane still works.
    """
    p = Path(jsonl_path)
    recs: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not recs:
        return {"dir": str(p.parent), "manifest": {}, "turns": [], "messages": []}
    term_block = (recs[-1].get("terminal") or {}) if recs else {}
    manifest: dict = dict(term_block.get("manifest") or {})
    # Surface the audit-only totals at the manifest level so the viewer's
    # "outcome / wall / tokens" header line has them.
    manifest.setdefault("outcome", term_block.get("outcome", "?"))
    manifest.setdefault("wall_clock_seconds", term_block.get("wall_clock_seconds"))
    manifest["total_tokens_in"] = term_block.get("total_tokens_in", 0)
    manifest["total_tokens_out"] = term_block.get("total_tokens_out", 0)
    turns = []
    messages: list[dict] = []
    # System prompt lives on the first turn only.
    sysp = recs[0].get("system_prompt")
    if sysp:
        messages.append({"role": "system", "content": sysp})
    for r in recs:
        # Turn record — translate to legacy keys the viewer recognises.
        png_rel = r.get("minimap_png")
        png_abs = str(p.parent / png_rel) if png_rel else None
        sig = r.get("signals") or {}
        turns.append(
            {
                "turn": r.get("turn"),
                "tick": r.get("tick"),
                "interrupt": r.get("interrupt"),
                "commands": r.get("commands_issued") or [],
                "ascii_minimap": (r.get("obs") or {}).get("minimap", ""),
                "signals": sig,
                "units": (r.get("obs") or {}).get("units_summary", []),
                "enemies": (r.get("obs") or {}).get("enemy_summary", []),
                "goal": {},  # not captured by FullPlayback
                "minimap_png": png_abs,
            }
        )
        if r.get("briefing"):
            messages.append({"role": "user", "content": r["briefing"]})
        resp = r.get("model_response") or {}
        if resp.get("text") or resp.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.get("text") or "",
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {
                                "name": (tc or {}).get("name", ""),
                                "arguments": (tc or {}).get("arguments", {}),
                            },
                        }
                        for i, tc in enumerate(resp.get("tool_calls") or [])
                    ],
                    "reasoning": resp.get("reasoning", ""),
                }
            )
    return {
        "dir": str(p.parent),
        "manifest": manifest,
        "turns": turns,
        "messages": messages,
    }


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 — partial/running episode
        return default


def _assistant_turns(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") == "assistant"]


def render_streamlit(root: str) -> None:  # pragma: no cover - UI glue
    import streamlit as st

    st.set_page_config(page_title="OpenRA-Bench playback", layout="wide")
    eps = find_episodes(root)
    if not eps:
        st.error(f"no episodes under {root}")
        return
    pick = st.sidebar.selectbox(
        "episode", eps, format_func=lambda p: f"{p.parent.name}/{p.name}"
    )
    ep = load_episode(pick)
    m = ep["manifest"]
    st.title(f"{m.get('scenario', pick)} — {m.get('outcome', '?')}")
    st.caption(
        f"{m.get('model','?')} · run {m.get('run_id','?')} · "
        f"turns {m.get('turns')}/{m.get('max_turns')} · "
        f"capability {m.get('capability')} · seed {m.get('seed')}"
    )

    # System prompt once at top (the deterministic scenario knowledge
    # the model was given) — same as the training pipeline viewer.
    sysp = next(
        (x.get("content", "") for x in ep["messages"]
         if x.get("role") == "system"), ""
    )
    if isinstance(sysp, list):
        sysp = " ".join(
            p.get("text", "") for p in sysp if isinstance(p, dict)
        )
    if sysp:
        with st.expander(f"🧠 System prompt ({len(sysp)} chars)",
                         expanded=False):
            st.code(sysp, language="text")

    users = [x for x in ep["messages"] if x.get("role") == "user"]
    asst = _assistant_turns(ep["messages"])
    for i, t in enumerate(ep["turns"]):
        with st.expander(
            f"turn {t['turn']} · tick {t.get('tick')}"
            + (f" · ⚡{t['interrupt']}" if t.get("interrupt") else ""),
            expanded=(i == 0),
        ):
            left, right = st.columns([1, 1])
            with left:
                if t.get("minimap_png"):
                    st.image(t["minimap_png"], caption="minimap")
                elif t.get("ascii_minimap"):
                    st.code(t["ascii_minimap"])
                st.json(t.get("signals", {}))
            with right:
                a = asst[i] if i < len(asst) else {}
                if a.get("reasoning"):
                    st.markdown("**reasoning**")
                    st.write(a["reasoning"])
                st.markdown("**commands**")
                st.code("\n".join(t.get("commands", [])) or "(none)")
                g = t.get("goal") or {}
                if g:
                    st.markdown(
                        f"**objective progress: "
                        f"{g.get('objective_progress', 0):.0%}**"
                        + ("  ✅ won" if g.get("won") else "")
                    )
                    for leaf in g.get("leaves", []):
                        st.progress(
                            min(1.0, float(leaf.get("ratio", 0.0))),
                            text=f"{leaf['name']} "
                            f"{leaf.get('current')}/{leaf.get('target')}",
                        )
                    rv = g.get("reward_vector", {})
                    st.caption("reward vector: " + "  ".join(
                        f"{k}={v:.2f}" for k, v in rv.items()
                    ))
