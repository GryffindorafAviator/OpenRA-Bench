"""Phase 3 — the 1v1 full-macro adversarial harness.

Drives TWO Controllers (Phase 1 contract) — any mix of LLM / human /
scripted — over ONE shared episode. Each side is fed its own
fog-of-war observation and issues commands into the same engine frame
via the `step_1v1` two-player command channel (engine change shipped
alongside this module). Full macro game: economy, production, tech and
combat all in play; the episode ends when one base falls or the turn
cap is hit.

`step_1v1` builds each side's orders independently (scoped to that
player's unit ownership) and applies them into the SAME first frame, so
neither side moves "first". It returns each player's observation from
its own shroud — so an LLM driving the enemy sees exactly the fogged
view its opponent sees, not a god's-eye board.

Usage:

    from openra_bench.one_v_one import run_1v1
    result = run_1v1(scenario_path, agent_controller, enemy_controller)
    print(result.winner, result.turns)

The two controllers are interchangeable: pass two `ModelAgent`s for
model-vs-model, a `ModelAgent` + a scripted Controller for
agent-vs-bot, or a `HumanController` on either side.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .controller import EpisodeContext, as_controller
from .rust_adapter import RustObsAdapter


@dataclass
class OneVOneResult:
    """Outcome of a 1v1 match.

    `winner` is from the agent side's frame of reference: "agent",
    "enemy", or "draw". `reason` records why the match ended."""

    winner: str  # "agent" | "enemy" | "draw"
    reason: str
    turns: int
    ticks: int
    agent_name: str
    enemy_name: str
    agent_trace: list[dict] = field(default_factory=list)
    enemy_trace: list[dict] = field(default_factory=list)


def _alive(render_state: dict) -> bool:
    """A side is still in the game if it has any unit or any building."""
    units = render_state.get("units_summary") or []
    buildings = render_state.get("own_buildings") or []
    return bool(units) or bool(buildings)


def _kills(render_state: dict) -> int:
    """Cumulative enemy units killed — the primary military-progress metric."""
    return int(render_state.get("units_killed", 0) or 0)


def _own_buildings(render_state: dict) -> int:
    """Count of own buildings still standing — the secondary military metric."""
    return len(render_state.get("own_buildings") or [])


def _economy_value(render_state: dict) -> int:
    """Cash + stored resources — the final-fallback economic metric."""
    return int(render_state.get("cash", 0) or 0) + int(
        render_state.get("resources", 0) or 0
    )


def _render_side_minimap(render_state, base_map, explored_set, *, level: str):
    """Render the same minimap PNG the model would see for ONE side's
    render_state. Returns base64-encoded PNG bytes or None. Never
    raises — playback persistence must NEVER break the engine run."""
    try:
        from .minimap import terrain_png_for
        from .prompt_v2 import minimap_b64 as _v2_mm

        terrain = terrain_png_for(base_map) if base_map else None
        png = _v2_mm(
            render_state, terrain, explored_set,
            constant_colors=level in ("easy", "medium"),
        )
        if png is None:
            from .agent import _render_minimap_b64
            png = _render_minimap_b64(render_state, terrain)
        return png
    except Exception:  # noqa: BLE001 — playback never breaks a run
        return None


def run_1v1(
    scenario_path: str,
    agent_controller: Any,
    enemy_controller: Any,
    seed: int = 0,
    max_turns: int = 200,
    progress: bool = True,
    playback_root=None,
    *,
    cell: str | None = None,
    half: str | None = None,
    run_id: str | None = None,
    agent_model: str | None = None,
    enemy_model: str | None = None,
    base_map: str | None = None,
    level: str = "easy",
) -> OneVOneResult:
    """Run one full-macro 1v1 match and return the result.

    `agent_controller` / `enemy_controller` are each a Controller, a
    `ModelAgent`, or a bare `agent_fn` callable — coerced through
    `as_controller()`. The scenario should leave the enemy side
    externally controlled (no `enemy.bot_type`); if it declares an
    engine bot, that bot co-drives the enemy actors alongside this
    harness's enemy controller.

    When `playback_root` is given, two `Playback` instances are
    created under it — `<playback_root>/agent_side/seed<N>/` and
    `<playback_root>/enemy_side/seed<N>/` — each capturing its
    respective controller's per-turn `render_state`, commands,
    signals, minimap PNG, and (for LLM controllers carrying a
    `.history` attribute) the full chat transcript. The caller is
    responsible for assembling the cell/half/run-id portion of the
    path; this function only owns the per-side leaf split. Manifest +
    score.json are written on both sides at episode end so the
    Playback UI can replay either perspective.
    """
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    agent = as_controller(agent_controller)
    enemy = as_controller(enemy_controller)

    pool = RustEnvPool(size=1, scenario_path=scenario_path)
    env = pool.acquire()
    try:
        agent.reset(
            EpisodeContext(seed=seed, side="agent", max_turns=max_turns)
        )
        enemy.reset(
            EpisodeContext(seed=seed, side="enemy", max_turns=max_turns)
        )

        agent_ad = RustObsAdapter()
        enemy_ad = RustObsAdapter()
        Command = env.Command

        # Seed each side's first fog-of-war observation WITHOUT stepping:
        # reset() gives the agent's; enemy_observation() gives the
        # enemy's at the same tick-0 state. (A two-player idle bootstrap
        # step would waste a whole decision turn — fatal on a combat map
        # where forces start in contact.) NOTE: the pool rebuilds the
        # underlying `_env` on reset(), so the raw env must be fetched
        # AFTER reset() — fetching it earlier captures a stale env.
        agent_ad.observe(env.reset(seed=seed))
        raw = getattr(env, "_env", None)
        if raw is None or not hasattr(raw, "step_1v1"):
            raise RuntimeError(
                "engine wheel lacks step_1v1 — rebuild the wheel "
                "(maturin develop --release) to run 1v1 matches"
            )
        enemy_ad.observe(raw.enemy_observation())

        agent_trace: list[dict] = []
        enemy_trace: list[dict] = []
        turns = 0
        done = False
        winner = "draw"
        reason = "turn cap reached"

        # Two controllers decide concurrently: both observe the same
        # tick-N state (captured BEFORE either `.act()` runs), then
        # their independent `.act()` calls go in parallel so the
        # per-turn wall-clock is max(agent_latency, enemy_latency)
        # rather than the sum — load-bearing when both controllers
        # are LLMs with multi-second round-trips. The engine's
        # `step_1v1` already applies both sides' orders in the same
        # frame so neither side moves "first"; this just stops the
        # bench harness from serializing the round-trips.
        executor = ThreadPoolExecutor(max_workers=2)

        # Per-turn full Playback persistence. When the caller passes a
        # playback_root, both sides get a sibling Playback under it so
        # the standard bench replay UI works on 1v1 episodes exactly
        # as on single-player ones (turns.jsonl + messages.json +
        # minimap_turn*.png + manifest.json + score.json). The earlier
        # ad-hoc `agent_history.json` / `enemy_history.json` route is
        # gone — messages.json (written by Playback.write_messages) is
        # the canonical LLM transcript.
        import json as _json
        import time as _time
        from pathlib import Path as _Path

        _episode_t0 = _time.monotonic()
        _agent_pb = None
        _enemy_pb = None
        _pb_root = None
        if playback_root:
            _pb_root = _Path(playback_root)
            _pb_root.mkdir(parents=True, exist_ok=True)
            try:
                from .playback import Playback

                _agent_pb = Playback(_pb_root, "agent_side", seed)
                _agent_pb.run_id = run_id
                _agent_pb.model = agent_model
                _enemy_pb = Playback(_pb_root, "enemy_side", seed)
                _enemy_pb.run_id = run_id
                _enemy_pb.model = enemy_model
            except Exception:  # noqa: BLE001 — playback never breaks a run
                _agent_pb = None
                _enemy_pb = None

        # Accumulating per-side fog history so the minimap shows the
        # explored-cells overlay correctly (same shape as the model's
        # input image, mirroring eval_core's `_pb_explored`).
        _agent_explored: set = set()
        _enemy_explored: set = set()

        def _accumulate_explored(rs: dict, store: set) -> None:
            try:
                obs = rs.get("_raw") or {}
                for cell_xy in obs.get("explored_cells") or []:
                    if isinstance(cell_xy, (list, tuple)) and len(cell_xy) == 2:
                        store.add((int(cell_xy[0]), int(cell_xy[1])))
            except Exception:  # noqa: BLE001
                pass

        def _flush_progress(turn_n: int) -> None:
            """Live rate metrics — same shape as the legacy
            progress.json the monitor reads. Writes under both side
            dirs so either tail target works."""
            if not (_agent_pb or _enemy_pb):
                return
            try:
                elapsed = _time.monotonic() - _episode_t0
                tps = (turn_n / elapsed) if elapsed > 0 else 0.0
                eta_s = ((max_turns - turn_n) / tps) if tps > 0 else 0.0
                row = {
                    "turn": turn_n,
                    "max_turns": max_turns,
                    "seed": seed,
                    "elapsed_s": round(elapsed, 1),
                    "turns_per_second": round(tps, 3),
                    "sec_per_turn": round(1.0 / tps, 2) if tps > 0 else None,
                    "eta_s": round(eta_s, 1),
                }
                if _agent_pb is not None:
                    (_agent_pb.dir / "progress.json").write_text(
                        _json.dumps(row)
                    )
                if _enemy_pb is not None:
                    (_enemy_pb.dir / "progress.json").write_text(
                        _json.dumps(row)
                    )
            except Exception:  # noqa: BLE001
                pass

        def _flush_messages(ctrl, pb) -> None:
            """If the controller exposes a `.history` attribute (the
            ModelAgent contract — system / user / assistant / tool
            messages with the minimap data-URL), dump the full
            transcript to `messages.json`. Scripted controllers don't
            carry a history and are silently skipped.

            FOOTGUN: `_build_1v1_controller` returns
            `ModelAgent(...).agent_fn` — a BOUND METHOD, not the
            instance. `as_controller(bound_method)` wraps it in a
            `FunctionController` (inherits from `BaseController` which
            has `self.history = []`). A naive `getattr(ctrl, "history")`
            picks up that empty inherited list instead of the real
            ModelAgent's history. The fix: unwrap via `.source` (the
            bound method's `__self__` that `FunctionController.__init__`
            captures) or `introspection_source(ctrl)`."""
            if pb is None:
                return
            # Prefer the underlying source (real ModelAgent) for
            # FunctionController-wrapped bound methods.
            src = getattr(ctrl, "source", None) or ctrl
            hist = getattr(src, "history", None)
            # Fallback: if source has no history, try the wrapper's
            # inherited list (will be `[]` for scripted; we still write
            # it so the file exists with the canonical empty shape).
            if hist is None:
                hist = getattr(ctrl, "history", None)
            if isinstance(hist, list):
                try:
                    pb.write_messages(hist)
                except Exception:  # noqa: BLE001
                    pass

        try:
            for turns in range(1, max_turns + 1):
                a_rs = agent_ad.render_state()
                e_rs = enemy_ad.render_state()

                a_fut = executor.submit(agent.act, a_rs, Command)
                e_fut = executor.submit(enemy.act, e_rs, Command)
                a_cmds = a_fut.result() or [Command.observe()]
                e_cmds = e_fut.result() or [Command.observe()]
                a_obs, e_obs, done, _info = raw.step_1v1(a_cmds, e_cmds)
                agent_ad.observe(a_obs, done=done)
                enemy_ad.observe(e_obs, done=done)

                # Playback: snapshot the POST-step render_state (what
                # the controller will see next turn) on both sides.
                if _agent_pb is not None or _enemy_pb is not None:
                    a_rs_post = agent_ad.render_state()
                    e_rs_post = enemy_ad.render_state()
                    _accumulate_explored(a_rs_post, _agent_explored)
                    _accumulate_explored(e_rs_post, _enemy_explored)
                    if _agent_pb is not None:
                        a_png = _render_side_minimap(
                            a_rs_post, base_map, _agent_explored, level=level
                        )
                        try:
                            _agent_pb.record_turn(
                                turns, a_rs_post, a_cmds,
                                agent_ad.signals, a_png,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        _flush_messages(agent, _agent_pb)
                    if _enemy_pb is not None:
                        e_png = _render_side_minimap(
                            e_rs_post, base_map, _enemy_explored, level=level
                        )
                        try:
                            _enemy_pb.record_turn(
                                turns, e_rs_post, e_cmds,
                                enemy_ad.signals, e_png,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        _flush_messages(enemy, _enemy_pb)
                    _flush_progress(turns)

                agent_trace.append(
                    {
                        "turn": turns,
                        "tick": agent_ad.signals.game_tick,
                        "n_cmds": len(a_cmds),
                    }
                )
                enemy_trace.append(
                    {
                        "turn": turns,
                        "tick": enemy_ad.signals.game_tick,
                        "n_cmds": len(e_cmds),
                    }
                )
                if progress:
                    a_rs_now = agent_ad.render_state()
                    e_rs_now = enemy_ad.render_state()
                    a_units = len(a_rs_now.get("units_summary") or [])
                    e_units = len(e_rs_now.get("units_summary") or [])
                    a_blds = _own_buildings(a_rs_now)
                    e_blds = _own_buildings(e_rs_now)
                    a_kills = _kills(a_rs_now)
                    e_kills = _kills(e_rs_now)
                    a_cash = _economy_value(a_rs_now)
                    e_cash = _economy_value(e_rs_now)
                    print(
                        f"[1v1 t{turns:>3} tick{agent_ad.signals.game_tick:>5}] "
                        f"agent: units={a_units} bld={a_blds} kills={a_kills} $={a_cash} | "
                        f"enemy: units={e_units} bld={e_blds} kills={e_kills} $={e_cash} "
                        f"(a_cmds={len(a_cmds)} e_cmds={len(e_cmds)})",
                        file=sys.stderr,
                        flush=True,
                    )
                if done:
                    break
        finally:
            executor.shutdown(wait=False)

        # Decide the winner from the final boards.
        a_rs = agent_ad.render_state()
        e_rs = enemy_ad.render_state()
        agent_alive = _alive(a_rs)
        enemy_alive = _alive(e_rs)
        if agent_alive and not enemy_alive:
            winner, reason = "agent", "enemy base eliminated"
        elif enemy_alive and not agent_alive:
            winner, reason = "enemy", "agent base eliminated"
        elif not agent_alive and not enemy_alive:
            winner, reason = "draw", "mutual elimination"
        else:
            # Both standing — deadline / turn cap. Layered tie-break:
            # 1) kills (who took down more of the opponent's force),
            # 2) own buildings remaining (military capital),
            # 3) economy (cash + resources) as the final tiebreak.
            # Each layer is consulted only if the previous is tied.
            ak, ek = _kills(a_rs), _kills(e_rs)
            if ak > ek:
                winner, reason = "agent", f"deadline — agent ahead on kills ({ak} vs {ek})"
            elif ek > ak:
                winner, reason = "enemy", f"deadline — enemy ahead on kills ({ek} vs {ak})"
            else:
                ab, eb = _own_buildings(a_rs), _own_buildings(e_rs)
                if ab > eb:
                    winner, reason = "agent", f"deadline — kills tied at {ak}; agent ahead on buildings ({ab} vs {eb})"
                elif eb > ab:
                    winner, reason = "enemy", f"deadline — kills tied at {ak}; enemy ahead on buildings ({eb} vs {ab})"
                else:
                    av, ev = _economy_value(a_rs), _economy_value(e_rs)
                    if av > ev:
                        winner, reason = "agent", f"deadline — kills+buildings tied; agent ahead on economy ({av} vs {ev})"
                    elif ev > av:
                        winner, reason = "enemy", f"deadline — kills+buildings tied; enemy ahead on economy ({ev} vs {av})"
                    else:
                        winner, reason = "draw", f"deadline — fully tied (kills={ak}, buildings={ab}, economy={av})"

        # Episode-completion writes: manifest.json + score.json on both
        # sides, plus a final transcript + progress flush. The
        # rate-history.jsonl append is preserved (one row per
        # completed episode, the episodes/min source-of-truth that the
        # monitor reads).
        if _agent_pb is not None or _enemy_pb is not None:
            _flush_progress(turns)
            try:
                ep_elapsed = _time.monotonic() - _episode_t0
                # Agent-side outcome (win/loss/draw) is from the agent
                # POV; enemy-side outcome is the mirror.
                if winner == "agent":
                    a_outcome, e_outcome = "win", "loss"
                elif winner == "enemy":
                    a_outcome, e_outcome = "loss", "win"
                else:
                    a_outcome, e_outcome = "draw", "draw"

                def _manifest(side_outcome: str, side_label: str,
                              side_model: str | None) -> dict:
                    return {
                        "mode": "1v1",
                        "scenario_path": str(scenario_path),
                        "cell": cell,
                        "half": half,
                        "seed": seed,
                        "side": side_label,
                        "model": side_model,
                        "run_id": run_id,
                        "outcome": side_outcome,
                        "winner": winner,
                        "reason": reason,
                        "turns": turns,
                        "ticks": agent_ad.signals.game_tick,
                        "episode_seconds": round(ep_elapsed, 1),
                        "agent_name": getattr(agent, "name", "agent"),
                        "enemy_name": getattr(enemy, "name", "enemy"),
                    }

                def _score(side_outcome: str) -> dict:
                    """Minimal score.json so the resume-gate + journal
                    cross-check sees a consistent on-disk record. The
                    1v1 mode doesn't run the scenario-grade composite
                    (perception/reasoning/action) — those depend on
                    a win_condition predicate, which 1v1 deliberately
                    stubs out — so we emit only the load-bearing
                    `outcome` field plus the raw winner/reason."""
                    return {
                        "outcome": side_outcome,
                        "winner": winner,
                        "reason": reason,
                        "composite": (
                            1.0 if side_outcome == "win"
                            else 0.0 if side_outcome == "loss"
                            else 0.5
                        ),
                    }

                if _agent_pb is not None:
                    _flush_messages(agent, _agent_pb)
                    _agent_pb.finalize(_manifest(a_outcome, "agent", agent_model))
                    (_agent_pb.dir / "score.json").write_text(
                        _json.dumps(_score(a_outcome), indent=2)
                    )
                if _enemy_pb is not None:
                    _flush_messages(enemy, _enemy_pb)
                    _enemy_pb.finalize(_manifest(e_outcome, "enemy", enemy_model))
                    (_enemy_pb.dir / "score.json").write_text(
                        _json.dumps(_score(e_outcome), indent=2)
                    )

                # Cell-level rate history (one row per completed
                # episode). Written at the playback_root parent so a
                # tailer can watch ALL seeds of a cell in one file.
                rate_row = {
                    "winner": winner,
                    "reason": reason,
                    "turns": turns,
                    "seed": seed,
                    "episode_seconds": round(ep_elapsed, 1),
                    "turns_per_second": (
                        round(turns / ep_elapsed, 3) if ep_elapsed > 0 else 0.0
                    ),
                    "completed_at": _time.time(),
                }
                try:
                    # Aggregate at the cell-level dir (one row per
                    # completed seed). The previous layout buried this
                    # under the seed dir so each file held a single
                    # row; hoisting up one level makes "episodes/min
                    # for this cell" a one-file tail.
                    (_pb_root / "rate-history.jsonl").open("a").write(
                        _json.dumps(rate_row) + "\n"
                    )
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001 — never break the run
                pass
        return OneVOneResult(
            winner=winner,
            reason=reason,
            turns=turns,
            ticks=agent_ad.signals.game_tick,
            agent_name=getattr(agent, "name", "agent"),
            enemy_name=getattr(enemy, "name", "enemy"),
            agent_trace=agent_trace,
            enemy_trace=enemy_trace,
        )
    finally:
        pool.release(env)
        pool.shutdown()
