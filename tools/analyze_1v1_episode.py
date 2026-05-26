"""Turn-by-turn analysis of a completed 1v1 episode.

Usage:
    python3 tools/analyze_1v1_episode.py <playback_episode_dir>

where the episode dir is the parent of agent_side/ and enemy_side/, e.g.
    data/runs/v1.1-prod-1v1/<pair>/<model>/1v1/playback/<run>/<cell>_<half>/

Output: a markdown report with kill timeline, build orders, key inflection
points, and notable LLM assistant turns (early/midgame/late tool calls).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Building actor types — used to split unit list into units vs buildings.
BLD = {
    "fact", "proc", "powr", "weap", "tent", "barr", "hpad", "afld",
    "silo", "fix", "gun", "pbox", "hbox", "sam", "ftur", "atek", "stek",
    "dome", "spen", "syrd", "apwr",
}


def _load_actor_costs() -> dict[str, int]:
    """Parse `<ACTOR>: Valued: Cost: N` from the embedded RA YAML."""
    rd = Path("/Users/berta/Projects/OpenRA-Rust/openra-data/src/embedded/rules")
    costs = {}
    if not rd.exists():
        return costs
    for yf in rd.glob("*.yaml"):
        for m in re.finditer(
            r"^([A-Z0-9_\.]+):\n((?:\t.+\n?)+)", yf.read_text(), re.MULTILINE
        ):
            actor = m.group(1).lower().split(".")[0]
            cm = re.search(r"Cost:\s*(\d+)", m.group(2))
            if cm and actor not in costs:
                costs[actor] = int(cm.group(1))
    return costs


def _classify(actors: list) -> tuple[Counter, Counter]:
    """Split a render_state actor list into (units, buildings) by type."""
    units = Counter()
    blds = Counter()
    for a in actors or []:
        if not isinstance(a, dict):
            continue
        typ = str(a.get("type") or a.get("actor_type") or "?").lower()
        if typ in BLD:
            blds[typ] += 1
        else:
            units[typ] += 1
    return units, blds


def _diff_actors(prev_list, cur_list) -> list[dict]:
    """Actors present in prev but NOT in cur — i.e. died/disappeared.
    Match by `id` (engine actor id) when available."""
    if not prev_list:
        return []
    cur_ids = {a.get("id") or a.get("actor_id") for a in (cur_list or []) if isinstance(a, dict)}
    lost = []
    for a in prev_list:
        if not isinstance(a, dict):
            continue
        aid = a.get("id") or a.get("actor_id")
        if aid is not None and aid not in cur_ids:
            lost.append(a)
    return lost


def _digest_assistant(content) -> str:
    """Compress an assistant message to ~200 chars."""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str):
        return ""
    return content[:200].replace("\n", " ")


def _summarize_tool_calls(msg) -> list[str]:
    """Extract a short list of tool-call summaries from an assistant message."""
    out = []
    for c in (msg.get("tool_calls") or []):
        fn = c.get("function", {})
        name = fn.get("name", "?")
        args = fn.get("arguments", "")
        if isinstance(args, dict):
            args = json.dumps(args)
        if isinstance(args, str):
            args = args[:120]
        out.append(f"{name}({args})")
    return out


def _digest_reasoning(msg) -> str:
    """Some models (gpt-5.x) put their chain-of-thought in `reasoning`."""
    r = msg.get("reasoning", "")
    if isinstance(r, list):
        r = " ".join(p.get("text", "") for p in r if isinstance(p, dict))
    return str(r or "")[:300].replace("\n", " ")


def analyze_side(side_dir: Path, costs: dict[str, int]) -> dict:
    """Analyze ONE side of a 1v1 episode (agent_side or enemy_side)."""
    tj = side_dir / "turns.jsonl"
    mj = side_dir / "messages.json"
    sj = side_dir / "score.json"
    ma = side_dir / "manifest.json"
    if not tj.exists():
        return {"error": f"no turns.jsonl at {tj}"}

    turns = []
    with tj.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                turns.append(json.loads(line))
            except Exception:
                continue

    score = json.loads(sj.read_text()) if sj.exists() else {}
    manifest = json.loads(ma.read_text()) if ma.exists() else {}
    messages = []
    if mj.exists():
        try:
            messages = json.loads(mj.read_text())
        except Exception:
            pass

    # Per-turn snapshot timeline + kill/loss diff
    timeline = []
    prev_units = None
    prev_enemies = None
    total_killed_value = 0
    total_lost_value = 0
    kill_log = []  # (turn, actor_type, cost)
    loss_log = []
    for t in turns:
        sigs = t.get("signals") or {}
        u, b = _classify(t.get("units"))
        eu, eb = _classify(t.get("enemies"))
        timeline.append({
            "turn": t.get("turn"),
            "cash": sigs.get("cash", 0),
            "kills": sigs.get("units_killed", 0),
            "u": sum(u.values()),
            "b": sum(b.values()),
            "eu": sum(eu.values()),
            "eb": sum(eb.values()),
        })
        # Diff: actors that DIED this turn
        for dead in _diff_actors(prev_units, t.get("units")):
            typ = str(dead.get("type") or dead.get("actor_type") or "?").lower()
            c = costs.get(typ, 0)
            total_lost_value += c
            loss_log.append((t.get("turn"), typ, c))
        for dead in _diff_actors(prev_enemies, t.get("enemies")):
            typ = str(dead.get("type") or dead.get("actor_type") or "?").lower()
            c = costs.get(typ, 0)
            total_killed_value += c
            kill_log.append((t.get("turn"), typ, c))
        prev_units = t.get("units") or []
        prev_enemies = t.get("enemies") or []

    # Sample assistant turns (early/middle/late)
    asst = [m for m in messages if m.get("role") == "assistant"]
    samples = []
    if asst:
        idxs = sorted(set([0, len(asst)//4, len(asst)//2, 3*len(asst)//4, len(asst)-1]))
        for i in idxs:
            m = asst[i]
            samples.append({
                "idx": i + 1,
                "of": len(asst),
                "text": _digest_assistant(m.get("content")),
                "reasoning": _digest_reasoning(m),
                "tool_calls": _summarize_tool_calls(m),
            })

    return {
        "score": score,
        "manifest": manifest,
        "timeline": timeline,
        "total_turns": len(turns),
        "kills_total_value": total_killed_value,
        "losses_total_value": total_lost_value,
        "kill_log": kill_log,
        "loss_log": loss_log,
        "n_messages": len(messages),
        "n_assistant_turns": len(asst),
        "asst_samples": samples,
    }


def render_md(ep_dir: Path) -> str:
    costs = _load_actor_costs()
    agent_dir = ep_dir / "agent_side" / "seed1"
    enemy_dir = ep_dir / "enemy_side" / "seed1"
    a = analyze_side(agent_dir, costs) if agent_dir.exists() else {"error": "no agent_side"}
    e = analyze_side(enemy_dir, costs) if enemy_dir.exists() else {"error": "no enemy_side"}
    rh = (ep_dir / "rate-history.jsonl")
    rate = {}
    if rh.exists():
        for line in rh.read_text().splitlines():
            try:
                rate = json.loads(line)
                break
            except Exception:
                pass

    pair = ep_dir.parts[-4] if len(ep_dir.parts) >= 4 else "?"  # v1.1-prod-1v1/<pair>/...
    cell_half = ep_dir.name

    out = [f"# 1v1 Episode Analysis: {cell_half}", ""]
    out.append(f"**Pair:** `{pair}`  ")
    out.append(f"**Cell:** `{a.get('manifest',{}).get('cell','?')}`  ")
    out.append(f"**Half:** `{a.get('manifest',{}).get('half','?')}`  ")
    out.append(f"**Seed:** `{a.get('manifest',{}).get('seed','?')}`  ")
    out.append(f"**Agent model:** `{a.get('manifest',{}).get('model','?')}`  ")
    out.append(f"**Enemy model:** `{e.get('manifest',{}).get('model','?')}`  ")
    out.append("")
    out.append(f"## Outcome")
    out.append(f"- **Winner**: `{rate.get('winner','?')}`")
    out.append(f"- **Reason**: {rate.get('reason','?')}")
    out.append(f"- **Turns**: {rate.get('turns','?')}/200")
    out.append(f"- **Duration**: {rate.get('episode_seconds','?')}s")
    out.append("")
    out.append(f"## Value-weighted scoreboard")
    out.append(f"| Side | Destroyed (enemy value $) | Lost (own value $) | Net |")
    out.append(f"|---|---|---|---|")
    a_dest = a.get("kills_total_value", 0)
    a_lost = a.get("losses_total_value", 0)
    e_dest = e.get("kills_total_value", 0)
    e_lost = e.get("losses_total_value", 0)
    out.append(f"| Agent | ${a_dest} | ${a_lost} | ${a_dest - a_lost:+} |")
    out.append(f"| Enemy | ${e_dest} | ${e_lost} | ${e_dest - e_lost:+} |")
    out.append("")

    # Trajectory: every ~20 turns
    out.append(f"## Trajectory (agent side, every 25 turns)")
    out.append("| turn | cash | kills_count | own_u | own_b | seen_e_u | seen_e_b |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|")
    for tl in (a.get("timeline") or [])[::25]:
        out.append(
            f"| {tl['turn']} | ${tl['cash']} | {tl['kills']} | "
            f"{tl['u']} | {tl['b']} | {tl['eu']} | {tl['eb']} |"
        )
    # Always include final turn
    if a.get("timeline"):
        last = a["timeline"][-1]
        out.append(
            f"| **{last['turn']}** | **${last['cash']}** | **{last['kills']}** | "
            f"**{last['u']}** | **{last['b']}** | **{last['eu']}** | **{last['eb']}** |"
        )
    out.append("")

    # Top destroyed enemy types
    out.append(f"## Agent's most-destroyed enemy types (by value)")
    by_type_a = defaultdict(int)
    for _t, typ, c in (a.get("kill_log") or []):
        by_type_a[typ] += c
    if by_type_a:
        out.append("| Type | Total destroyed value | # kills |")
        out.append("|---|---:|---:|")
        for typ, v in sorted(by_type_a.items(), key=lambda x: -x[1])[:6]:
            n = sum(1 for _, t2, _ in (a.get("kill_log") or []) if t2 == typ)
            out.append(f"| `{typ}` | ${v} | {n} |")
    else:
        out.append("_(no actor-id-based kills detected — render_state may lack `id` field)_")
    out.append("")

    # Notable assistant turns
    out.append(f"## Agent LLM commentary (sampled across the episode, {a.get('n_assistant_turns',0)} total)")
    for s in a.get("asst_samples", [])[:5]:
        out.append(f"### Turn {s['idx']}/{s['of']}")
        if s["text"]:
            out.append(f"**Text:** {s['text']}")
        if s.get("reasoning"):
            out.append(f"**Reasoning:** {s['reasoning']}")
        if s["tool_calls"]:
            out.append("**Tools called:** " + ", ".join(f"`{tc}`" for tc in s["tool_calls"][:3]))
        out.append("")

    out.append(f"## Files in this episode")
    out.append(f"- `agent_side/seed1/turns.jsonl`: {a.get('total_turns',0)} turn records")
    out.append(f"- `agent_side/seed1/messages.json`: {a.get('n_messages',0)} messages ({a.get('n_assistant_turns',0)} assistant turns)")
    out.append(f"- `enemy_side/seed1/turns.jsonl`: {e.get('total_turns',0)} turn records")
    out.append(f"- `enemy_side/seed1/messages.json`: {e.get('n_messages',0)} messages")
    minimaps = list(agent_dir.glob("minimap_turn*.png")) if agent_dir.exists() else []
    out.append(f"- agent-side minimap PNGs: {len(minimaps)} (turns {1}–{len(minimaps)})")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ep = Path(sys.argv[1])
    print(render_md(ep))
