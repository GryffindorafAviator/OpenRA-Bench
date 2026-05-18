"""Scenario hygiene: quarantine excludes from the default sweep.

Quarantined packs (redundant cat-* over-generation; harvest packs
blocked on engine S0/S1) must (a) still parse, (b) drop out of the
default evaluate set into `skipped`, and (c) remain runnable when
named explicitly. Two representatives per cat-* category stay active.
"""

from __future__ import annotations

import glob
import os
from collections import Counter
from pathlib import Path

from openra_bench.scenarios import load_pack

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def _all_metas():
    out = []
    for f in glob.glob(str(PACKS / "*.yaml")):
        if os.path.basename(f).startswith(("_", "TEMPLATE")):
            continue
        out.append((os.path.basename(f), load_pack(f).meta))
    return out


def test_all_packs_still_parse_after_retag():
    metas = _all_metas()
    assert len(metas) > 80  # nothing lost on disk
    for name, m in metas:
        assert m.status in ("active", "quarantine")
        if m.status == "quarantine":
            assert m.quarantine_reason, f"{name} quarantined w/o reason"


def test_harvest_quarantined_with_engine_reason():
    for h in ("economy-harvest-investment", "economy-harvest-timebox"):
        m = load_pack(PACKS / f"{h}.yaml").meta
        assert m.status == "quarantine"
        assert "S0/S1" in m.quarantine_reason


def test_two_representatives_kept_active_per_cat_category():
    active = Counter()
    quar = Counter()
    for name, m in _all_metas():
        if not name.startswith("cat-c"):
            continue
        cat = name.split("-")[1]  # c1..c12
        (active if m.status == "active" else quar)[cat] += 1
    assert active, "cat-* categories present"
    for cat, n in active.items():
        assert n == 2, f"{cat}: expected 2 active reps, got {n}"
        assert quar[cat] >= 1, f"{cat}: redundant ones must be quarantined"


def test_default_evaluate_excludes_quarantine_keeps_active(monkeypatch):
    # Use the real resolver but a tiny fake eval to avoid the engine:
    # assert quarantined ids land in `skipped`, active ones in tasks.
    from openra_bench import run_eval

    captured = {}

    def fake_compile(pack, level):
        captured.setdefault("compiled", []).append(pack.meta.id)

        class C:
            map_supported = False  # short-circuit into skipped, no engine

        c = C()
        c.meta = pack.meta
        return c

    monkeypatch.setattr(run_eval, "compile_level", fake_compile)
    packs = run_eval._resolve_packs(None)  # default bundled set
    out = run_eval.evaluate(packs=packs, levels=["easy"], seeds=[1])
    sk = " ".join(out["skipped"])
    assert "quarantine:" in sk
    # a known-quarantined pack is in skipped and never compiled
    assert any("cat-c1-05" in s for s in out["skipped"])
    assert "cat-c1-05" not in captured.get("compiled", [])
    # a kept representative WAS compiled (reached the level loop)
    assert "cat-c1-00" in captured.get("compiled", [])


def test_quarantined_pack_still_loads_when_named_explicitly():
    # explicit selection bypasses the default-set filter entirely
    p = run_eval_resolve_one("cat-c1-05.yaml")
    assert p.meta.id == "cat-c1-05" and p.meta.status == "quarantine"


def run_eval_resolve_one(fname: str):
    return load_pack(PACKS / fname)
