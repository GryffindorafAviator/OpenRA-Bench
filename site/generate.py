#!/usr/bin/env python3
"""Generate static scenario data for the mission-player website.

Maintainer-only preprocessing step. Reads every active scenario pack,
renders map thumbnails, generates bilingual (EN/ZH) human-friendly
descriptions deterministically (no AI API), and writes:

    public/scenarios.json   — full scenario metadata
    public/maps/<id>.png    — per-pack map thumbnail

Usage:
    python site/generate.py            # generate all
    python site/generate.py --dry-run  # print counts, don't write
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PUBLIC = Path(__file__).resolve().parent / "public"
MAPS_DIR = PUBLIC / "maps"

# ── Bilingual description generation (deterministic, no AI API) ────────

# Capability descriptions
_CAP_ZH = {
    "perception": "感知",
    "reasoning": "推理",
    "action": "行动",
    "adversarial": "对抗",
}

_CAP_DESC_EN = {
    "perception": "Tests the agent's ability to observe and interpret the battlefield.",
    "reasoning": "Tests the agent's ability to plan and make strategic decisions.",
    "action": "Tests the agent's ability to execute commands precisely.",
    "adversarial": "Tests the agent's ability to compete against a reactive opponent.",
}

_CAP_DESC_ZH = {
    "perception": "测试智能体观察和理解战场的能力。",
    "reasoning": "测试智能体规划和做出战略决策的能力。",
    "action": "测试智能体精确执行命令的能力。",
    "adversarial": "测试智能体与反应型对手竞争的能力。",
}

_DIFF_ZH = {"easy": "简单", "medium": "中等", "hard": "困难"}


_translate_cache: dict[str, str] = {}


def _google_translate_zh(text: str) -> str:
    """Translate English text to Chinese via Google Translate API."""
    if not text or not text.strip():
        return text
    if text in _translate_cache:
        return _translate_cache[text]

    import time
    import urllib.parse
    import urllib.request

    url = (
        "https://translate.googleapis.com/translate_a/single"
        "?client=gtx&sl=en&tl=zh-CN&dt=t&q="
        + urllib.parse.quote(text)
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json as _json
            data = _json.loads(resp.read().decode())
            result = "".join(seg[0] for seg in data[0] if seg[0])
        result = _fixup_game_terms(result)
        _translate_cache[text] = result
        time.sleep(0.15)
        return result
    except Exception as e:
        print(f"  [warn] Google Translate failed: {e}")
        return text


def _fixup_game_terms(zh: str) -> str:
    """Fix game-specific terms that Google Translate consistently gets wrong."""
    fixes = [
        ("游戏勾选", "游戏刻"),
        ("游戏滴答", "游戏刻"),
        ("游戏刻度", "游戏刻"),
        ("游戏蜱虫", "游戏刻"),
        ("游戏壁虱", "游戏刻"),
        ("游戏报价", "游戏刻"),
        ("决策轮次", "决策回合"),
        ("决策转弯", "决策回合"),
        ("决策回合", "决策回合"),
        ("勾号", "刻"),
    ]
    for wrong, right in fixes:
        zh = zh.replace(wrong, right)
    return zh


def _translate_objective_zh(en: str) -> str:
    """Translate a full objective text to Chinese.

    Canonicalises the load-bearing objective prefixes ("WIN WHEN:" /
    "YOU LOSE IF:") deterministically — the Google Translate output
    drifts across runs ("获胜时间" / "如果您失败" / "胜利条件" all
    appear depending on context) — then forwards the rest of each line
    to Google Translate so non-prefix wording stays natural.
    """
    if not en or not en.strip():
        return en
    out_lines: list[str] = []
    for line in en.split("\n"):
        out_lines.append(_translate_line_zh(line))
    return "\n".join(out_lines)


def _translate_line_zh(line: str) -> str:
    """Translate one line, with deterministic prefix replacement for
    the canonical objective markers."""
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    # Canonical prefixes: replace the English head with the canonical
    # Chinese head and only Google-translate the remainder. Mirrors the
    # objective_brief output in openra_bench/game_knowledge.py.
    PREFIXES = (
        ("WIN WHEN:", "胜利条件："),
        ("YOU LOSE IF:", "失败条件："),
    )
    for en_pref, zh_pref in PREFIXES:
        if stripped.startswith(en_pref):
            rest = stripped[len(en_pref):].strip()
            zh_rest = _google_translate_zh(rest) if rest else ""
            return f"{indent}{zh_pref}{zh_rest}"
    return _google_translate_zh(line) if line.strip() else line


def _annotator_hints_en(pack) -> list[str]:
    """Generate annotator-facing hints from pack metadata."""
    hints = []
    cap = pack.meta.capability
    hints.append(f"This scenario tests {cap} capability.")
    if pack.meta.benchmark_anchor:
        hints.append(f"Benchmark anchors: {', '.join(pack.meta.benchmark_anchor)}.")
    hints.append(f"Map: {pack.base_map if isinstance(pack.base_map, str) else 'procedurally generated'}.")
    return hints


def _annotator_hints_zh(pack) -> list[str]:
    """Generate annotator-facing hints in Chinese."""
    hints = []
    cap = _CAP_ZH.get(pack.meta.capability, pack.meta.capability)
    hints.append(f"此场景测试{cap}能力。")
    if pack.meta.benchmark_anchor:
        hints.append(f"基准锚点：{', '.join(pack.meta.benchmark_anchor)}。")
    map_name = pack.base_map if isinstance(pack.base_map, str) else "程序生成"
    hints.append(f"地图：{map_name}。")
    return hints


def build_scenario(pack, objective_brief_fn) -> dict:
    """Build the internal scenario model for one pack."""
    cap = pack.meta.capability
    levels = {}
    for lv_name in ("easy", "medium", "hard"):
        try:
            cl = pack.compile(lv_name)
            en_obj = objective_brief_fn(
                cl.scenario.description, cl.win_condition,
                cl.fail_condition, cl.max_turns,
                getattr(cl, "objective_coords", "exact"),
            )
            zh_obj = _translate_objective_zh(en_obj)
            levels[lv_name] = {
                "description": cl.scenario.description,
                "maxTurns": cl.max_turns,
                "startingCash": cl.starting_cash,
                "fogMode": getattr(cl, "fog_mode", "vision"),
                "objectiveCoords": getattr(cl, "objective_coords", "exact"),
                "objective": {"en": en_obj, "zh": zh_obj},
            }
        except Exception as e:
            levels[lv_name] = {"error": str(e)}

    anchors = ", ".join(pack.meta.benchmark_anchor) if pack.meta.benchmark_anchor else ""
    return {
        "scenarioId": pack.meta.id,
        "title": pack.meta.title,
        "capability": cap,
        "capabilityLabel": {"en": cap.title(), "zh": _CAP_ZH.get(cap, cap)},
        "map": {
            "type": "image",
            "src": f"maps/{pack.meta.id}.png",
            "baseMap": pack.base_map if isinstance(pack.base_map, str) else "generated",
        },
        "raw": {
            "realWorldMeaning": pack.meta.real_world_meaning,
            "roboticsAnalogue": pack.meta.robotics_analogue,
            "benchmarkAnchor": anchors,
            "author": pack.meta.author,
        },
        "humanReadable": {
            "summary": {
                "en": pack.meta.real_world_meaning,
                "zh": _translate_objective_zh(pack.meta.real_world_meaning),
            },
            "playerInstructions": {
                "en": [levels[lv].get("objective", {}).get("en", "") for lv in ("easy", "medium", "hard") if "objective" in levels.get(lv, {})],
                "zh": [levels[lv].get("objective", {}).get("zh", "") for lv in ("easy", "medium", "hard") if "objective" in levels.get(lv, {})],
            },
            "annotatorHints": {
                "en": _annotator_hints_en(pack),
                "zh": _annotator_hints_zh(pack),
            },
            "ambiguities": {"en": [], "zh": []},
        },
        "levels": levels,
    }


def render_map_thumbnail(pack, out_path: Path) -> bool:
    """Render a map thumbnail for one pack."""
    try:
        from openra_bench.scenarios.loader import compile_level
        cl = compile_level(pack, "easy")
        from openra_bench.minimap import render_tactical_minimap
        rs = {"minimap": "", "units_summary": [], "enemy_summary": []}
        # Try to get a real render state from a compiled level
        from openra_bench.eval_core import make_env
        env = make_env(cl, seed=1)
        obs = env.reset()
        from openra_bench.rust_adapter import RustObsAdapter
        adapter = RustObsAdapter(cl.scenario)
        rs = adapter.render_state(obs)
        env.close()
        img = render_tactical_minimap(rs, scale=3, grid=True, legend=False)
        if img:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path))
            return True
    except Exception:
        pass
    # Fallback: try minimap from scenario description alone
    try:
        from openra_bench.minimap import render_tactical_minimap
        cl = pack.compile("easy")
        actors = []
        for a in getattr(cl.scenario, 'actors', []) or []:
            if hasattr(a, 'position') and a.position:
                actors.append({
                    "cell_x": a.position[0] if isinstance(a.position, (list, tuple)) else getattr(a.position, 'x', 0),
                    "cell_y": a.position[1] if isinstance(a.position, (list, tuple)) else getattr(a.position, 'y', 0),
                    "type": getattr(a, 'actor_type', '?'),
                    "id": "0",
                })
        rs = {
            "minimap": "",
            "units_summary": actors[:10],
            "enemy_summary": [],
        }
        img = render_tactical_minimap(rs, scale=3, grid=True, legend=False)
        if img:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path))
            return True
    except Exception:
        pass
    return False


def generate(dry_run: bool = False) -> dict:
    """Generate all static data. Returns summary stats."""
    from openra_bench.game_knowledge import objective_brief
    from openra_bench.scenarios import discover_packs

    packs = [p for p in discover_packs() if p.meta.status == "active"]
    scenarios = []
    maps_ok = 0

    for p in packs:
        s = build_scenario(p, objective_brief)
        scenarios.append(s)
        if not dry_run:
            img_path = MAPS_DIR / f"{p.meta.id}.png"
            if render_map_thumbnail(p, img_path):
                maps_ok += 1

    has_en = sum(1 for s in scenarios if s["humanReadable"]["playerInstructions"]["en"])
    has_zh = sum(1 for s in scenarios if s["humanReadable"]["playerInstructions"]["zh"])

    stats = {
        "total": len(scenarios),
        "maps": maps_ok if not dry_run else "skipped",
        "english_instructions": has_en,
        "chinese_instructions": has_zh,
        "capabilities": {},
    }
    from collections import Counter
    for cap, n in Counter(s["capability"] for s in scenarios).items():
        stats["capabilities"][cap] = n

    if not dry_run:
        PUBLIC.mkdir(parents=True, exist_ok=True)
        out = PUBLIC / "scenarios.json"
        out.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out} ({len(scenarios)} scenarios)")
        print(f"Maps rendered: {maps_ok}/{len(scenarios)}")
    else:
        print(f"[dry-run] Would generate {len(scenarios)} scenarios")

    print(f"English instructions: {has_en}/{len(scenarios)}")
    print(f"Chinese instructions: {has_zh}/{len(scenarios)}")
    for cap, n in sorted(stats["capabilities"].items()):
        print(f"  {cap}: {n}")

    return stats


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    generate(dry_run=dry)
