"""Leaderboard data layer: ingest, ranking, capability breakdown,
min-episode gate, determinism, tie-breaking. Pure/file-backed — no
Gradio."""

from __future__ import annotations

from openra_bench.leaderboard import build_table, ingest_run


def _stats(win_rate, comp, p, r, a, n=10, caps=("perception", "reasoning")):
    eps = []
    for i in range(n):
        cap = caps[i % len(caps)]
        eps.append(
            {
                "cell": f"{cap}-x:easy",
                "capability": cap,
                "outcome": "win" if i < round(win_rate * n) else "loss",
                "composite": comp,
                "perception": p,
                "reasoning": r,
                "action": a,
                "weakest_link": "reasoning",
            }
        )
    return {
        "overall": {
            "n": n,
            "win_rate": win_rate,
            "composite_mean": comp,
            "perception_mean": p,
            "reasoning_mean": r,
            "action_mean": a,
            "weakest_link_hist": {"reasoning": n},
        },
        "summary": {"perception-x:easy": {}, "reasoning-x:easy": {}},
        "episodes": eps,
    }


def test_ingest_and_rank(tmp_path):
    s = tmp_path / "lb.jsonl"
    ingest_run(_stats(0.4, 0.40, 0.5, 0.3, 0.9), "model-A", s)
    ingest_run(_stats(0.7, 0.72, 0.8, 0.7, 0.9), "model-B", s)
    table = build_table(s)
    assert [r["model"] for r in table] == ["model-B", "model-A"]
    assert table[0]["rank"] == 1 and table[1]["rank"] == 2
    assert table[0]["weakest_link"] == "reasoning"
    # per-capability breakdown surfaced
    assert set(table[0]["by_capability"]) <= {"perception", "reasoning", "action"}


def test_best_run_per_model_kept(tmp_path):
    s = tmp_path / "lb.jsonl"
    ingest_run(_stats(0.3, 0.30, 0.3, 0.3, 0.3), "m", s)
    ingest_run(_stats(0.6, 0.65, 0.6, 0.6, 0.6), "m", s)  # better
    ingest_run(_stats(0.5, 0.50, 0.5, 0.5, 0.5), "m", s)
    table = build_table(s)
    assert len(table) == 1
    assert table[0]["composite"] == 0.65


def test_min_episode_gate(tmp_path):
    s = tmp_path / "lb.jsonl"
    ingest_run(_stats(1.0, 0.9, 0.9, 0.9, 0.9, n=3), "tiny", s)  # < MIN
    assert build_table(s) == []
    assert build_table(s, min_episodes=1)[0]["model"] == "tiny"


def test_ranking_is_deterministic_and_breaks_ties(tmp_path):
    s = tmp_path / "lb.jsonl"
    # Equal composite → higher win_rate ranks first, then name.
    ingest_run(_stats(0.5, 0.50, 0.5, 0.5, 0.5), "z-model", s)
    ingest_run(_stats(0.8, 0.50, 0.5, 0.5, 0.5), "a-model", s)
    t1 = [r["model"] for r in build_table(s)]
    t2 = [r["model"] for r in build_table(s)]
    assert t1 == t2 == ["a-model", "z-model"]


def test_handles_missing_store(tmp_path):
    assert build_table(tmp_path / "nope.jsonl") == []
