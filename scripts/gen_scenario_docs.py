"""Generate docs/scenarios.html — the human-readable scenario catalog.

For every ACTIVE pack: title, capability, why-it-exists, the runnable
configs (or 3 levels), and per cell the plain-language objective the
model actually sees (objective_brief: description + WIN/LOSE + turn
budget). Run:  python scripts/gen_scenario_docs.py [--open]
"""

from __future__ import annotations

import glob
import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKS = ROOT / "openra_bench" / "scenarios" / "packs"
OUT = ROOT / "docs" / "scenarios.html"

sys.path.insert(0, str(ROOT))  # runnable as a standalone script

from openra_bench.game_knowledge import objective_brief  # noqa: E402
from openra_bench.scenarios import load_pack  # noqa: E402

_CAP_COLOR = {
    "perception": "#7497db", "reasoning": "#9b8cce",
    "action": "#5fae7a", "adversarial": "#d2683c",
}


def _esc(s) -> str:
    return html.escape(str(s)).replace("\n", "<br>")


def _cells(pack):
    """[(label, CompiledLevel)] — configs if declared, else 3 levels."""
    out = []
    if pack.configs:
        for c in pack.configs:
            out.append((c.name, pack.compile_config(c.name)))
    else:
        for lv in ("easy", "medium", "hard"):
            out.append((lv, pack.compile(lv)))
    return out


def build() -> str:
    packs = []
    for f in sorted(glob.glob(str(PACKS / "*.yaml"))):
        b = os.path.basename(f)
        if b.startswith(("_", "TEMPLATE")):
            continue
        p = load_pack(f)
        if p.meta.status == "active":
            packs.append(p)

    by_cap: dict[str, list] = {}
    for p in packs:
        by_cap.setdefault(p.meta.capability, []).append(p)

    parts = [
        "<!doctype html><meta charset=utf-8>",
        "<title>OpenRA-Bench — Scenario Catalog</title>",
        """<style>
        body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
             margin:0;background:#0f1115;color:#e6e6e6}
        header{padding:24px 32px;background:#161922;
               border-bottom:1px solid #2a2f3a}
        h1{margin:0;font-size:22px} .sub{color:#9aa3b2;margin-top:6px}
        main{padding:24px 32px;max-width:1100px}
        h2{margin:34px 0 8px;font-size:18px;border-bottom:1px solid #2a2f3a;
           padding-bottom:6px}
        .pack{background:#161922;border:1px solid #2a2f3a;border-radius:10px;
              padding:16px 18px;margin:14px 0}
        .ptitle{font-size:17px;font-weight:600}
        .pid{color:#7e8796;font-size:12px;font-family:ui-monospace,monospace}
        .cap{display:inline-block;padding:2px 9px;border-radius:10px;
             color:#fff;font-size:12px;margin-left:8px;vertical-align:middle}
        .why{color:#c3cad6;margin:8px 0 12px;font-size:14px}
        .cell{border-left:3px solid #2a2f3a;padding:6px 0 6px 14px;
               margin:10px 0}
        .clab{font-weight:600;color:#cdd5e3}
        pre{white-space:pre-wrap;background:#0f1115;border:1px solid #242a35;
            border-radius:6px;padding:10px 12px;margin:6px 0 0;
            font:13px/1.45 ui-monospace,monospace;color:#d7dce6}
        .toc a{color:#7497db;text-decoration:none;margin-right:14px}
        </style>""",
        "<header><h1>OpenRA-Bench — Scenario Catalog</h1>",
        f"<div class=sub>{len(packs)} active scenarios · the title, why "
        "it exists, and the exact objective the model is given per "
        "runnable config.</div>",
        "<div class=sub toc>" + " ".join(
            f"<a href='#{c}'>{c} ({len(v)})</a>"
            for c, v in sorted(by_cap.items())
        ) + "</div></header><main>",
    ]

    for cap in sorted(by_cap):
        parts.append(f"<h2 id='{cap}'>{cap}</h2>")
        for p in sorted(by_cap[cap], key=lambda x: x.meta.id):
            col = _CAP_COLOR.get(cap, "#666")
            parts.append("<div class=pack>")
            parts.append(
                f"<div><span class=ptitle>{_esc(p.meta.title)}</span>"
                f"<span class=cap style='background:{col}'>{cap}</span>"
                f"<div class=pid>{p.meta.id} · {p.base_map}</div></div>"
            )
            parts.append(
                f"<div class=why><b>Why:</b> {_esc(p.meta.real_world_meaning)}"
                f"<br><b>Robotics analogue:</b> "
                f"{_esc(p.meta.robotics_analogue)}</div>"
            )
            try:
                cells = _cells(p)
            except Exception as e:  # noqa: BLE001
                parts.append(f"<div class=why>(compile error: {_esc(e)})</div>")
                cells = []
            for label, cl in cells:
                fog = getattr(cl, "fog_mode", "vision")
                ob = objective_brief(
                    cl.scenario.description, cl.win_condition,
                    cl.fail_condition, cl.max_turns,
                )
                parts.append(
                    f"<div class=cell><span class=clab>{label}</span> "
                    f"<span class=pid>(level {cl.level} · fog "
                    f"{fog})</span><pre>{_esc(ob)}</pre></div>"
                )
            parts.append("</div>")
    parts.append("</main>")
    return "".join(parts)


def main(argv):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
    if "--open" in argv:
        import subprocess

        subprocess.run(["open", str(OUT)], check=False)


if __name__ == "__main__":
    main(sys.argv[1:])
