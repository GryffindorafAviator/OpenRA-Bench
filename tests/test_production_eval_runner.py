"""Tests for `tools/run_production_eval.py` — the campaign orchestrator.

These tests don't launch `run_eval` (that would require a live model
endpoint). Instead they fabricate a small run-dir tree that mirrors the
real layout, then exercise the read-only paths:

  - `load_manifest` is idempotent (creates → reread returns same shape)
  - `cell_status` reads `eval_stats.json` + `status.json` correctly
  - `render_status_table` runs cleanly across all 12 cells
  - `summarise_run` / `render_summary_md` produce the expected sections
  - `cmd_status` (the CLI subcommand) prints a non-empty table

The whole orchestrator is import-only safe (no side effects on import).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = REPO_ROOT / "tools" / "run_production_eval.py"


def _load_runner_module():
    """Load `tools/run_production_eval.py` as a module without polluting
    `tools/` with an `__init__.py`."""
    spec = importlib.util.spec_from_file_location(
        "run_production_eval", str(TOOL_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_production_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner_module()


def _fake_eval_stats(*, model: str, n_wins: int = 2, n_loss: int = 1,
                     n_draw: int = 0,
                     extra_episodes: list[dict] | None = None) -> dict:
    eps: list[dict] = []
    for i in range(n_wins):
        eps.append({
            "cell": f"combat-test:easy",
            "capability": "action",
            "outcome": "win",
            "composite": 0.8,
            "weakest_link": "perception",
            "seed": 1 + i,
        })
    for i in range(n_loss):
        eps.append({
            "cell": f"econ-buildup:hard",
            "capability": "economy",
            "outcome": "loss",
            "composite": 0.2,
            "weakest_link": "reasoning",
            "seed": 1 + i,
        })
    for i in range(n_draw):
        eps.append({
            "cell": f"adv-rps-counter-pick:easy",
            "capability": "adversarial",
            "outcome": "draw",
            "composite": 0.4,
            "weakest_link": "objective",
            "seed": 1 + i,
        })
    if extra_episodes:
        eps.extend(extra_episodes)
    n = len(eps)
    return {
        "run_id": "20260525-000000",
        "model": model,
        "overall": {
            "n": n,
            "win_rate": n_wins / n if n else 0.0,
            "composite_mean": sum(e["composite"] for e in eps) / n if n else 0.0,
        },
        "summary": {},
        "episodes": eps,
    }


def _seed_cell(prod_dir: Path, slug: str, type_: str, *,
               state: str, stats: dict | None = None) -> None:
    cell = RUNNER.cell_dir(prod_dir, slug, type_)
    cell.mkdir(parents=True, exist_ok=True)
    if stats is not None:
        (cell / "eval_stats.json").write_text(json.dumps(stats))
    (cell / "status.json").write_text(json.dumps({
        "model_slug": slug, "type": type_, "state": state,
    }))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_manifest_creates_then_idempotent(tmp_path: Path) -> None:
    mf1 = RUNNER.load_manifest(tmp_path)
    assert mf1["schema"] == "v1"
    assert mf1["campaign"] == "v1.1-prod"
    # one cell per (model, type)
    assert len(mf1["cells"]) == len(RUNNER.MODELS) * len(RUNNER.TYPES)
    # spot-check known model
    assert "qwen3.5-9b:scenarios" in mf1["cells"]
    assert mf1["cells"]["qwen3.5-9b:scenarios"]["state"] == "not_started"

    # second call must NOT overwrite (started_at preserved)
    mf2 = RUNNER.load_manifest(tmp_path)
    assert mf2["started_at"] == mf1["started_at"]


def test_cell_status_reads_eval_stats(tmp_path: Path) -> None:
    RUNNER.load_manifest(tmp_path)
    stats = _fake_eval_stats(model="Qwen/Qwen3.5-9B")
    _seed_cell(tmp_path, "qwen3.5-9b", "scenarios",
               state="complete", stats=stats)
    s = RUNNER.cell_status(tmp_path, "qwen3.5-9b", "scenarios")
    assert s["state"] == "complete"
    assert s["cells_done"] == 3  # 2 wins + 1 loss
    assert 0.6 < s["win_rate"] < 0.7  # 2/3
    assert s["composite_mean"] is not None
    assert s["last_cell"] is not None


def test_render_status_table_covers_all_cells(tmp_path: Path) -> None:
    RUNNER.load_manifest(tmp_path)
    table = RUNNER.render_status_table(tmp_path)
    # header + separator + 12 rows
    lines = table.splitlines()
    assert lines[0].startswith("MODEL")
    expected = len(RUNNER.MODELS) * len(RUNNER.TYPES) + 2
    assert len(lines) == expected, (
        f"expected {expected} lines (header+sep+{expected-2} cells), "
        f"got {len(lines)}")


def test_summarise_run_per_family_and_top_bottom(tmp_path: Path) -> None:
    stats = _fake_eval_stats(
        model="Qwen/Qwen3.5-9B", n_wins=4, n_loss=2, n_draw=1,
        extra_episodes=[
            {"cell": "scout-fog-recover:medium",
             "capability": "perception",
             "outcome": "win", "composite": 0.7,
             "weakest_link": "perception", "seed": 4},
            {"cell": "build-power-online-first:easy",
             "capability": "build",
             "outcome": "loss", "composite": 0.1,
             "weakest_link": "action", "seed": 5},
        ])
    summary = RUNNER.summarise_run(stats, slug="qwen3.5-9b",
                                   type_="scenarios")
    # core counters
    assert summary["episodes"] == 9
    assert summary["wins"] == 5
    assert summary["losses"] == 3
    assert summary["draws"] == 1

    fams = {r["family"] for r in summary["per_family"]}
    # combat-test → combat-micro; econ-buildup → economy;
    # adv-rps → tempo-strategy; scout-fog → scout-perception;
    # build-power → build-tech-power
    assert "family1-combat-micro" in fams
    assert "family2-economy" in fams
    assert "family9-tempo-strategy" in fams
    assert "family4-scout-perception" in fams
    assert "family6-build-tech-power" in fams

    assert summary["best_cells"]  # non-empty
    assert summary["worst_cells"]  # non-empty
    # composite descending in best_cells
    comps = [r["composite_mean"] for r in summary["best_cells"]]
    assert comps == sorted(comps, reverse=True)


def test_render_summary_md_includes_required_sections(tmp_path: Path) -> None:
    stats = _fake_eval_stats(model="Qwen/Qwen3.5-9B")
    summary = RUNNER.summarise_run(stats, slug="qwen3.5-9b",
                                   type_="scenarios")
    md = RUNNER.render_summary_md(summary)
    assert "# qwen3.5-9b / scenarios" in md
    assert "per family" in md
    assert "top 5 cells" in md
    # no spawn-rotation block in scenarios mode
    assert "spawn rotation" not in md


def test_render_summary_md_1v1_includes_spawn(tmp_path: Path) -> None:
    stats = _fake_eval_stats(
        model="Qwen/Qwen3.5-9B",
        extra_episodes=[
            {"cell": "adversarial-1v1-macro:easy",
             "capability": "adversarial",
             "outcome": "win", "composite": 0.6,
             "weakest_link": "objective", "seed": 1,
             "spawn_point": "NW"},
            {"cell": "adversarial-1v1-macro:easy",
             "capability": "adversarial",
             "outcome": "loss", "composite": 0.3,
             "weakest_link": "objective", "seed": 2,
             "spawn_point": "SE"},
        ])
    summary = RUNNER.summarise_run(stats, slug="qwen3.5-9b", type_="1v1")
    md = RUNNER.render_summary_md(summary)
    assert "spawn rotation" in md
    assert "spawn-NW" in md or "spawn-SE" in md


def test_pack_family_classification() -> None:
    assert RUNNER.pack_family("combat-foo") == "family1-combat-micro"
    assert RUNNER.pack_family("econ-bar") == "family2-economy"
    assert RUNNER.pack_family("def-base") == "family3-defense"
    assert RUNNER.pack_family("scout-fog") == "family4-scout-perception"
    assert RUNNER.pack_family("lh-marathon") == "family5-long-horizon"
    assert RUNNER.pack_family("build-power") == "family6-build-tech-power"
    assert RUNNER.pack_family("rob-cash") == "family7-procedure-robustness"
    assert RUNNER.pack_family("mfb-two") == "family8-multi-front-coord"
    assert RUNNER.pack_family("adv-rps") == "family9-tempo-strategy"
    assert RUNNER.pack_family("spec-thief") == "family10-special-misc"


def test_update_cell_state_writes_manifest_and_status(tmp_path: Path) -> None:
    RUNNER.load_manifest(tmp_path)
    cell = RUNNER.update_cell_state(
        tmp_path, "qwen3.5-9b", "scenarios",
        state="in_progress", concurrency=20)
    assert cell["state"] == "in_progress"
    assert cell["concurrency"] == 20
    # status.json mirrors the cell
    sp = RUNNER.cell_status_path(tmp_path, "qwen3.5-9b", "scenarios")
    assert sp.exists()
    snap = json.loads(sp.read_text())
    assert snap["state"] == "in_progress"


def test_status_subcommand_runs_without_error(tmp_path: Path,
                                              capsys) -> None:
    # The `status` CLI must run cleanly on a fresh dir.
    rc = RUNNER.main(["status", "--prod-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MODEL" in out
    assert "qwen3.5-9b" in out


def test_campaign_gates_1v1_on_scenarios_complete(tmp_path: Path,
                                                  monkeypatch) -> None:
    """`campaign` must NOT launch a 1v1 cell while its scenarios cell is
    still incomplete. We monkeypatch `launch_cell` to a recorder, then
    assert only scenarios cells were called."""
    RUNNER.load_manifest(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_launch(prod_dir, slug, type_, **kw):
        calls.append((slug, type_))
        # leave the cell as "not_started" so the next slug also gets a
        # scenarios call; the goal is to verify NO 1v1 call slips in
        # while scenarios are incomplete.
        return 1  # non-zero → not marked complete

    monkeypatch.setattr(RUNNER, "launch_cell", fake_launch)

    import argparse
    args = argparse.Namespace(
        prod_dir=str(tmp_path), concurrency=20,
        levels="easy", seeds="1", opponent="scripted:stall",
        auto_pr=False, dry_run=False,
    )
    RUNNER.cmd_campaign(args)

    # Every recorded launch must be a scenarios call.
    assert calls, "expected at least one scenarios launch attempt"
    types_called = {t for _, t in calls}
    assert types_called == {"scenarios"}, (
        f"1v1 launched before scenarios completed: {calls}")
