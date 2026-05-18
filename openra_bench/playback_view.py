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
    running episode (missing files become empty)."""
    d = Path(ep_dir)
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
    """All ``.../seed<N>`` episode dirs under a playback root."""
    return sorted(
        p.parent for p in Path(root).glob("**/manifest.json")
    ) or sorted(Path(root).glob("**/seed*"))


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
        f"turns {m.get('turns')}/{m.get('max_turns')} · "
        f"capability {m.get('capability')} · seed {m.get('seed')}"
    )

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
