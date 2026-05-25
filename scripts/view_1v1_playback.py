"""Streamlit 1v1 playback viewer — dual-side, full LLM output.

Shows both controllers' perspectives side-by-side per turn: each side's
fog-of-war minimap, the model's text + chain-of-thought reasoning + tool
calls (with arguments), and the engine signals snapshot.

Usage:
    pip install streamlit
    streamlit run scripts/view_1v1_playback.py -- <playback_root>

The playback root can be either:
  * the campaign root (e.g. data/runs/v1.1-prod-1v1) — the picker walks
    pair × model × run × cell_half combos and lets you pick any episode
  * a specific cell_half dir (e.g. .../adversarial-1v1-macro_easy_normal)
    — jumps straight into that episode

For each picked episode it pairs `agent_side/seed1/` with the sibling
`enemy_side/seed1/` automatically and renders both perspectives in
parallel columns.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st


def _read_json(p: Path, default):
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _digest(s, maxlen=2000) -> str:
    if isinstance(s, list):
        s = " ".join(p.get("text", "") for p in s if isinstance(p, dict))
    if not isinstance(s, str):
        return ""
    return s[:maxlen]


def _assistant_per_turn(messages: list[dict]) -> list[dict]:
    """Return the assistant message per turn (in order)."""
    return [m for m in messages if m.get("role") == "assistant"]


def _user_per_turn(messages: list[dict]) -> list[dict]:
    """Return the user observation message per turn."""
    return [m for m in messages if m.get("role") == "user"]


def _format_tool_call(tc: dict) -> str:
    fn = tc.get("function", {}) or {}
    name = fn.get("name", "?")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    arg_str = json.dumps(args, separators=(",", ":")) if isinstance(args, (dict, list)) else str(args)
    if len(arg_str) > 220:
        arg_str = arg_str[:220] + "…"
    return f"{name}({arg_str})"


def find_1v1_episodes(root: Path) -> list[Path]:
    """Find all 1v1 episode dirs (the cell_half level — parent of
    agent_side + enemy_side). Skips _archive unless it's the only data."""
    # An episode dir has both `agent_side/seed*/` and `enemy_side/seed*/`.
    # We walk and check for agent_side dirs.
    out: list[Path] = []
    for agent_side in root.rglob("agent_side"):
        if not agent_side.is_dir():
            continue
        ep_dir = agent_side.parent
        if (ep_dir / "enemy_side").is_dir():
            out.append(ep_dir)
    out.sort(key=lambda p: str(p))
    return out


def render_side(col, side_dir: Path, side_label: str, turn_idx: int):
    """Render one side's per-turn panel."""
    if not (side_dir / "seed1").exists():
        col.warning(f"no seed1 dir under {side_dir}")
        return
    seed_dir = side_dir / "seed1"
    turns = _read_jsonl(seed_dir / "turns.jsonl")
    messages = _read_json(seed_dir / "messages.json", [])
    manifest = _read_json(seed_dir / "manifest.json", {})

    asst = _assistant_per_turn(messages)
    users = _user_per_turn(messages)

    col.markdown(f"### {side_label}: `{manifest.get('model','?')}`")
    col.caption(f"outcome: **{manifest.get('outcome','?')}** · "
                f"reason: {manifest.get('reason','?')}")

    if turn_idx >= len(turns):
        col.info(f"turn {turn_idx + 1} not yet recorded for this side "
                 f"(side has {len(turns)} turns)")
        return
    t = turns[turn_idx]

    # Minimap
    mp = seed_dir / f"minimap_turn{(t.get('turn') or 0):03d}.png"
    if mp.exists():
        col.image(str(mp), caption=f"minimap turn {t.get('turn')}")
    else:
        col.caption(f"(no minimap_turn{(t.get('turn') or 0):03d}.png)")

    # Model output (the requested triple: text + reasoning + tool_calls)
    if turn_idx < len(asst):
        a = asst[turn_idx]
        content = _digest(a.get("content"))
        reasoning = _digest(a.get("reasoning"))
        tool_calls = a.get("tool_calls") or []
        if content:
            col.markdown("**model text output:**")
            col.write(content)
        if reasoning:
            col.markdown("**model reasoning (chain-of-thought):**")
            col.write(reasoning)
        if tool_calls:
            col.markdown(f"**tool calls ({len(tool_calls)}):**")
            for tc in tool_calls:
                col.code(_format_tool_call(tc), language="text")
    else:
        col.caption(f"(no assistant message recorded for turn {turn_idx + 1})")

    # Engine signals
    sigs = t.get("signals") or {}
    if sigs:
        col.markdown("**signals (state at end of turn):**")
        col.json(sigs)

    # Engine commands as recorded in turns.jsonl
    cmds = t.get("commands") or []
    if cmds:
        col.markdown("**engine-recorded commands:**")
        col.code("\n".join(str(c) for c in cmds), language="text")


def main(root_arg: str) -> None:
    st.set_page_config(page_title="OpenRA-Bench 1v1 playback", layout="wide")
    root = Path(root_arg).resolve()
    if not root.exists():
        st.error(f"playback root does not exist: {root}")
        return

    # Find all episodes
    episodes = find_1v1_episodes(root)
    if not episodes:
        st.error(f"no 1v1 episodes (with agent_side + enemy_side) under {root}")
        return

    # Sidebar episode picker
    def _label(ep: Path) -> str:
        # Compact label: <pair>/<cell_half>
        parts = ep.parts
        try:
            i = parts.index("v1.1-prod-1v1")
            pair = parts[i + 1] if i + 1 < len(parts) else "?"
        except ValueError:
            pair = "?"
        return f"{pair} / {ep.name}"

    pick = st.sidebar.selectbox(
        f"episode ({len(episodes)} found)", episodes, format_func=_label
    )

    # Episode-level summary
    agent_ma = _read_json(pick / "agent_side" / "seed1" / "manifest.json", {})
    enemy_ma = _read_json(pick / "enemy_side" / "seed1" / "manifest.json", {})
    rh_rows = _read_jsonl(pick / "rate-history.jsonl")
    rh = rh_rows[0] if rh_rows else {}

    st.title(f"1v1: {agent_ma.get('cell', pick.name)}")
    st.caption(
        f"agent: **{agent_ma.get('model','?')}** vs "
        f"enemy: **{enemy_ma.get('model','?')}** · "
        f"half: **{agent_ma.get('half','?')}** · seed: {agent_ma.get('seed','?')}"
    )

    if rh:
        st.markdown(
            f"**Winner**: `{rh.get('winner','?')}` · "
            f"**Reason**: {rh.get('reason','?')} · "
            f"**Turns**: {rh.get('turns','?')}/200 · "
            f"**Duration**: {rh.get('episode_seconds','?')}s"
        )

    # Find the number of turns to display (the shorter of the two sides
    # for in-flight episodes; the full for completed)
    agent_turns = _read_jsonl(pick / "agent_side" / "seed1" / "turns.jsonl")
    enemy_turns = _read_jsonl(pick / "enemy_side" / "seed1" / "turns.jsonl")
    n_turns = max(len(agent_turns), len(enemy_turns))
    if n_turns == 0:
        st.info("episode has no turn data yet")
        return

    st.markdown(f"**{n_turns} turn(s) recorded**")

    # Turn picker — session-state-backed so prev/next buttons and JS
    # arrow-key bindings can mutate the value.
    if "turn_n" not in st.session_state:
        st.session_state.turn_n = 1
    if st.session_state.turn_n > n_turns:
        st.session_state.turn_n = n_turns
    if st.session_state.turn_n < 1:
        st.session_state.turn_n = 1

    nav_cols = st.columns([1, 1, 6, 1, 1])
    if nav_cols[0].button("⏮ first", use_container_width=True):
        st.session_state.turn_n = 1
        st.rerun()
    if nav_cols[1].button("◀ prev", use_container_width=True):
        st.session_state.turn_n = max(1, st.session_state.turn_n - 1)
        st.rerun()
    nav_cols[2].markdown(f"### Turn **{st.session_state.turn_n}** / {n_turns}")
    if nav_cols[3].button("next ▶", use_container_width=True):
        st.session_state.turn_n = min(n_turns, st.session_state.turn_n + 1)
        st.rerun()
    if nav_cols[4].button("last ⏭", use_container_width=True):
        st.session_state.turn_n = n_turns
        st.rerun()

    # Slider for quick scrubbing
    turn_n = st.sidebar.slider(
        "turn (slider — use ← → keys on the main pane to step one at a time)",
        1, n_turns, st.session_state.turn_n,
    )
    if turn_n != st.session_state.turn_n:
        st.session_state.turn_n = turn_n
        st.rerun()

    # Arrow-key binding: dispatches synthetic clicks on the prev/next
    # buttons by finding them via their text content. Streamlit doesn't
    # have a first-party keyboard API, so a small bit of injected JS is
    # the cleanest cross-platform solution.
    st.components.v1.html(
        """
        <script>
        (function() {
          if (window.__1v1_keys_bound) return;
          window.__1v1_keys_bound = true;
          const findBtn = (label) => {
            const buttons = window.parent.document.querySelectorAll('button');
            for (const b of buttons) {
              const t = (b.innerText || '').trim();
              if (t.startsWith(label)) return b;
            }
            return null;
          };
          window.parent.document.addEventListener('keydown', (ev) => {
            // Ignore if user is typing in an input/textarea
            const tag = (ev.target && ev.target.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea') return;
            if (ev.key === 'ArrowLeft') {
              const b = findBtn('◀');
              if (b) { ev.preventDefault(); b.click(); }
            } else if (ev.key === 'ArrowRight') {
              const b = findBtn('next');
              if (b) { ev.preventDefault(); b.click(); }
            } else if (ev.key === 'Home') {
              const b = findBtn('⏮');
              if (b) { ev.preventDefault(); b.click(); }
            } else if (ev.key === 'End') {
              const b = findBtn('last');
              if (b) { ev.preventDefault(); b.click(); }
            }
          });
        })();
        </script>
        """,
        height=0,
    )

    turn_idx = st.session_state.turn_n - 1

    # Two-column dual-side render
    col_agent, col_enemy = st.columns(2)
    render_side(col_agent, pick / "agent_side", "AGENT side", turn_idx)
    render_side(col_enemy, pick / "enemy_side", "ENEMY side", turn_idx)

    # User observation (briefing) toggle below
    with st.expander("📜 Agent's user observation (briefing + minimap data-URL) for this turn",
                     expanded=False):
        users = _user_per_turn(_read_json(pick / "agent_side" / "seed1" / "messages.json", []))
        if turn_idx < len(users):
            u = users[turn_idx]
            c = u.get("content", "")
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict):
                        if p.get("type") == "text":
                            st.code(p.get("text", ""), language="markdown")
                        elif p.get("type") == "image_url":
                            url = (p.get("image_url") or {}).get("url", "")
                            if url.startswith("data:image"):
                                st.caption(f"(embedded image, {len(url)} chars)")
                                # base64 → image
                                b64 = url.split(",", 1)[1] if "," in url else ""
                                if b64:
                                    try:
                                        st.image(base64.b64decode(b64), caption="user-msg minimap")
                                    except Exception:
                                        pass
            else:
                st.code(str(c), language="markdown")

    # System prompt
    with st.sidebar.expander("🧠 System prompt (agent side)"):
        sysm = next((m for m in _read_json(pick / "agent_side" / "seed1" / "messages.json", [])
                     if m.get("role") == "system"), {})
        c = sysm.get("content", "")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        st.code(c[:4000], language="text")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "data/runs/v1.1-prod-1v1"
    main(arg)
