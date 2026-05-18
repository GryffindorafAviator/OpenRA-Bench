"""Battle viewer data layer (pure, unit-tested).

Indexes a playback root that may hold *many* runs/models/scenarios and
serves the cascade the UI needs:

    run_id  →  model  →  scenario (pack:level:split @ seed)  →  episode

plus per-turn assembly (`episode_view`) and the comparison pairing rule
(B is locked to A's scenario+seed; only run/model vary).

Layout written by run_eval:
    <root>/<run_id>__<model>/<pack:level:split>/seed<n>/{manifest,turns,
        messages,score,minimap_turnNNN}.

Identity is read from each manifest (run_id/model/scenario/seed), never
parsed from the path, so the folder scheme can change without breaking
the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .playback_view import load_episode


@dataclass(frozen=True)
class EpisodeRef:
    run_id: str
    model: str
    scenario: str  # pack:level:split
    seed: int
    outcome: str
    composite: float | None
    objective: float | None
    dir: str

    @property
    def scenario_key(self) -> str:
        """The (scenario, seed) identity comparison mode locks on."""
        return f"{self.scenario}@{self.seed}"


def scan(root: str | Path) -> list[EpisodeRef]:
    """Every episode under ``root``, newest run first. Tolerant of
    partial/running episodes (missing score → None)."""
    out: list[EpisodeRef] = []
    for man in Path(root).glob("**/manifest.json"):
        d = man.parent
        ep = load_episode(d)
        m = ep["manifest"]
        if not m:
            continue
        score = {}
        sp = d / "score.json"
        if sp.exists():
            import json

            try:
                score = json.loads(sp.read_text())
            except Exception:  # noqa: BLE001
                score = {}
        out.append(EpisodeRef(
            run_id=str(m.get("run_id") or "unknown"),
            model=str(m.get("model") or "agent"),
            scenario=str(m.get("scenario") or d.parent.name),
            seed=int(m.get("seed") or 0),
            outcome=str(m.get("outcome") or "?"),
            composite=score.get("composite"),
            objective=score.get(
                "objective_progress", m.get("objective_progress")
            ),
            dir=str(d),
        ))
    out.sort(key=lambda e: (e.run_id, e.model, e.scenario, e.seed),
             reverse=True)
    return out


def _u(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def runs(idx: list[EpisodeRef]) -> list[str]:
    return _u(e.run_id for e in idx)


def models(idx: list[EpisodeRef], run_id: str) -> list[str]:
    return _u(e.model for e in idx if e.run_id == run_id)


def scenarios(idx: list[EpisodeRef], run_id: str, model: str) -> list[str]:
    return _u(
        e.scenario_key
        for e in idx
        if e.run_id == run_id and e.model == model
    )


def find(idx: list[EpisodeRef], run_id: str, model: str,
         scenario_key: str) -> EpisodeRef | None:
    for e in idx:
        if (e.run_id == run_id and e.model == model
                and e.scenario_key == scenario_key):
            return e
    return None


def compare_candidates(idx: list[EpisodeRef],
                       a: EpisodeRef) -> list[EpisodeRef]:
    """Episodes runnable as B against A: same scenario+seed, any other
    run/model (A itself excluded)."""
    return [
        e for e in idx
        if e.scenario_key == a.scenario_key and e.dir != a.dir
    ]


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return content or ""


def _role_turns(messages: list[dict], role: str) -> list[dict]:
    return [m for m in messages if m.get("role") == role]


def episode_view(ep_dir: str | Path, turn_idx: int) -> dict:
    """Everything to render one turn of one episode. ``turn_idx`` is
    0-based and clamped to range."""
    ep = load_episode(ep_dir)
    turns = ep["turns"]
    n = len(turns)
    if n == 0:
        return {"n_turns": 0, "manifest": ep["manifest"]}
    i = max(0, min(turn_idx, n - 1))
    t = turns[i]
    users = _role_turns(ep["messages"], "user")
    assts = _role_turns(ep["messages"], "assistant")
    g = t.get("goal") or {}
    return {
        "n_turns": n,
        "turn_idx": i,
        "turn": t.get("turn"),
        "tick": t.get("tick"),
        "interrupt": t.get("interrupt"),
        "minimap_png": t.get("minimap_png"),
        "ascii_minimap": t.get("ascii_minimap", ""),
        "briefing": _text(users[i]["content"]) if i < len(users) else "",
        "reasoning": (assts[i].get("reasoning", "")
                      if i < len(assts) else ""),
        "assistant_text": (_text(assts[i].get("content"))
                           if i < len(assts) else ""),
        "commands": t.get("commands", []),
        "signals": t.get("signals", {}),
        "goal": g,
        "objective_progress": g.get("objective_progress"),
        "won": g.get("won"),
        "manifest": ep["manifest"],
    }
