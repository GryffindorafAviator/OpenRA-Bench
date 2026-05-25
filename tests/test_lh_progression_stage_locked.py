"""lh-progression-stage-locked pack — full no-cheat validation on Rust.

Wave-10 long-horizon REASONING pack: a PERT-style stage-locked
progression enforced by the Wave-2 `then:` happened-before composite.
The five-stage chain is:

    STAGE 1 (POWER):   has_building: powr
    STAGE 2 (INTAKE):  has_building: proc
    STAGE 3 (CAPITAL): economy_value_gte: M
    STAGE 4 (TECH):    has_building: weap
    STAGE 5 (ACTION):  units_killed_gte: K

Each stage gates the next, so skipping is impossible by construction:
the `then:` latch cannot advance past a stage that was never observed
true, and the chain never completes.

Bar (per CLAUDE.md — real LOSS, never DRAW): the intended ordered-
progression policy WINS on every (level, seed); stall, skip-a-stage
(never build powr / build but never engage), and a hunt-everything
brute (which razes a sentinel `fact`) all LOSE on every (level, seed).

Scenario shape:
  - rush-hour-arena, allies vs soviet, no scripted bot.
  - easy:   M=3500, K=2, starting_cash 3000, 70 turns.
  - medium: M=3000, K=3, starting_cash 2200, 70 turns.
  - hard:   M=3000, K=3, starting_cash 2400, ≥2 spawn_point groups.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-progression-stage-locked.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-level (M, K) — kept in lock-step with the YAML.
_M = {"easy": 3500, "medium": 3000, "hard": 3000}
_K = {"easy": 2, "medium": 3, "hard": 3}
# Mid-east kill cluster centre (well separated from the sentinel facts).
# Targets the cluster directly so the assault stops on contact rather
# than carrying past the cluster toward an unreachable / sentinel cell.
_TGT = {"easy": (60, 30), "medium": (60, 30), "hard": (60, 20)}


# ── Obs helpers ───────────────────────────────────────────────────


def _bset(obs):
    return {b.get("type") for b in (obs.get("own_buildings", []) or [])}


def _basecell(obs):
    for b in obs.get("own_buildings", []) or []:
        if b.get("type") == "fact":
            return b.get("cell_x"), b.get("cell_y")
    return 10, 18


def _harvest_cmd(obs, Cmd):
    """Kickstart the harv auto-cycle (harmless every turn)."""
    units = obs.get("units_summary", []) or []
    harv = [u["id"] for u in units if u.get("type") == "harv"]
    if not harv:
        return None
    h0 = next(u for u in units if u.get("type") == "harv")
    return Cmd.harvest(harv, 22, int(h0.get("cell_y", 18)))


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on every (level, seed). The pre-placed
    2tnk are stance:0 (HoldFire), so a staller can't win for free by
    the engine auto-delivering the army. The chain stays at stage 0
    and the episode times out past within_ticks."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _intended_policy(level: str):
    """The intended ordered progression — must WIN on every
    (level, seed):
      STAGE 1: build powr.
      STAGE 2: build proc (refinery — brings harvest income online).
      STAGE 3: wait for economy_value to accrue past M.
      STAGE 4: build weap (war factory).
      STAGE 5: set AttackAnything stance + attack-move the strike
               force onto the mid-east HOLD-FIRE kill cluster.
    The strike force stops once K kills land (the win fires), so it
    never drifts to a sentinel fact — no enemy building is razed."""
    M = _M[level]
    tx, ty = _TGT[level]
    stance_set: set = set()
    attacked: set = set()
    cap_hit = {"v": False}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        prod = [
            (p.get("type") if isinstance(p, dict) else p)
            for p in (obs.get("production", []) or [])
        ]
        cash = int(obs.get("cash", 0) or 0)
        ev = int(obs.get("economy_value", 0) or 0)
        blds = _bset(obs)
        bx, by = _basecell(obs)
        cmds = []
        h = _harvest_cmd(obs, Cmd)
        if h is not None:
            cmds.append(h)

        def build_stage(name, cost, dx, dy):
            """Queue `name` (when cash allows) and place it every turn
            it sits ready in the production queue. Returns True while
            this stage is still unfinished."""
            if name in blds:
                return False
            if name in prod:
                cmds.append(Cmd.place_building(name, bx + dx, by + dy))
                return True
            if cash >= cost:
                cmds.append(Cmd.build(name))
            return True

        # STAGE 1 — power.
        if build_stage("powr", 300, 4, 0):
            return cmds or [Cmd.observe()]
        # STAGE 2 — intake.
        if build_stage("proc", 1400, 2, 4):
            return cmds or [Cmd.observe()]
        # STAGE 3 — capital reserve (wait for harvest income). Once
        # the M reserve is observed at least once, the `then:` latch
        # has credited stage 3 and we proceed regardless of subsequent
        # spend-down (spending on weap drops the live ev back below M).
        if ev >= M:
            cap_hit["v"] = True
        if not cap_hit["v"]:
            return cmds or [Cmd.observe()]
        # STAGE 4 — tech.
        if build_stage("weap", 2000, 6, 4):
            return cmds or [Cmd.observe()]
        # STAGE 5 — terminal action: hunt the mid-east kill cluster.
        strike = [u for u in units if u.get("type") == "2tnk"]
        new = [u["id"] for u in strike if u["id"] not in stance_set]
        if new:
            cmds.append(Cmd.set_stance(new, 3))
            stance_set.update(new)
        fresh = [u["id"] for u in strike if u["id"] not in attacked]
        if fresh:
            cmds.append(Cmd.attack_move(fresh, tx, ty))
            attacked.update(fresh)
        return cmds or [Cmd.observe()]

    return pol


def _skip_powr_policy(level: str):
    """Try to skip STAGE 1 entirely — build proc directly. The engine
    refuses proc (prereq `anypower`), so STAGE 1 never latches and the
    `then:` chain stays at index 0. Must LOSE on every (level, seed)."""
    def pol(obs, Cmd):
        prod = [
            (p.get("type") if isinstance(p, dict) else p)
            for p in (obs.get("production", []) or [])
        ]
        cash = int(obs.get("cash", 0) or 0)
        blds = _bset(obs)
        bx, by = _basecell(obs)
        cmds = []
        h = _harvest_cmd(obs, Cmd)
        if h is not None:
            cmds.append(h)
        if "proc" not in blds:
            if "proc" in prod:
                cmds.append(Cmd.place_building("proc", bx + 2, by + 4))
            elif cash >= 1400:
                cmds.append(Cmd.build("proc"))
        return cmds or [Cmd.observe()]
    return pol


def _build_no_attack_policy(level: str):
    """Build the full powr → proc → weap economy but NEVER engage —
    STAGE 5 (units_killed_gte:K) never latches, the chain stalls at
    index 4. Must LOSE on every (level, seed)."""
    def pol(obs, Cmd):
        prod = [
            (p.get("type") if isinstance(p, dict) else p)
            for p in (obs.get("production", []) or [])
        ]
        cash = int(obs.get("cash", 0) or 0)
        blds = _bset(obs)
        bx, by = _basecell(obs)
        cmds = []
        h = _harvest_cmd(obs, Cmd)
        if h is not None:
            cmds.append(h)

        def build_stage(name, cost, dx, dy):
            if name in blds:
                return False
            if name in prod:
                cmds.append(Cmd.place_building(name, bx + dx, by + dy))
                return True
            if cash >= cost:
                cmds.append(Cmd.build(name))
            return True

        if build_stage("powr", 300, 4, 0):
            return cmds or [Cmd.observe()]
        if build_stage("proc", 1400, 2, 4):
            return cmds or [Cmd.observe()]
        build_stage("weap", 2000, 6, 4)
        return cmds or [Cmd.observe()]
    return pol


def _hunt_brute_policy(level: str):
    """A brute that skips the whole chain: flip every tank to
    AttackAnything from turn 0 and attack-move across the map. It
    hunts and razes the sentinel `fact` markers, tripping the
    `enemy_buildings_destroyed_gte` fail clause — a real LOSS, not a
    DRAW from the engine auto-`done`. Builds nothing, so the chain
    also never advances past stage 0. Must LOSE on every (level,
    seed)."""
    tx, ty = _TGT[level]
    stance_set: set = set()
    moved: set = set()

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        strike = [u for u in units if u.get("type") == "2tnk"]
        cmds = []
        new = [u["id"] for u in strike if u["id"] not in stance_set]
        if new:
            cmds.append(Cmd.set_stance(new, 3))
            stance_set.update(new)
        fresh = [u["id"] for u in strike if u["id"] not in moved]
        if fresh:
            cmds.append(Cmd.attack_move(fresh, tx, ty))
            moved.update(fresh)
        return cmds or [Cmd.observe()]
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ───────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-progression-stage-locked"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: PERT critical path / PlanBench
    staged dependencies / project management."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PERT critical path" in a for a in anchors), anchors
    assert any("PlanBench staged dependencies" in a for a in anchors), anchors
    assert any("project management" in a.lower() for a in anchors), anchors


def test_then_chain_is_five_stages():
    """The headline mechanic: every level's win is a 5-clause
    `then:` chain in the exact powr → proc → M → weap → kills order."""
    expected = [
        "has_building",
        "has_building",
        "economy_value_gte",
        "has_building",
        "units_killed_gte",
    ]
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        thens = [cl for cl in inner if "then" in cl]
        assert thens, f"{lvl} win missing then-chain: {win}"
        clauses = (thens[0]["then"] or {}).get("clauses") or []
        assert len(clauses) == 5, (
            f"{lvl} chain must have 5 stages; got {len(clauses)}"
        )
        keys = [next(iter(cl.keys())) for cl in clauses]
        assert keys == expected, f"{lvl} stage order wrong: {keys}"


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_fail_condition_punishes_razing_enemy_buildings():
    """Each level's fail must include enemy_buildings_destroyed_gte so
    a hunt-everything brute LOSES (instead of the engine auto-`done`
    collapsing the run to a DRAW). STAGE 5 is a unit-kill stage."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fail = c.fail_condition.model_dump(exclude_none=True)
        clauses = fail.get("any_of") or []
        assert any("enemy_buildings_destroyed_gte" in cl for cl in clauses), (
            f"{lvl} fail missing enemy_buildings_destroyed_gte: {fail}"
        )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (UPGRADED
    contract in tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_starting_cash_below_economy_value_bar():
    """M (STAGE 3) must be tuned ABOVE starting_cash so the capital
    reserve genuinely requires harvest income — proc must be built
    first, which makes STAGE 2 → STAGE 3 a real gate."""
    pack = load_pack(PACK)
    for lvl, m in _M.items():
        c = compile_level(pack, lvl)
        assert c.starting_cash < m, (
            f"{lvl} starting_cash={c.starting_cash} >= M={m} — STAGE 3 "
            f"could pass without harvest income"
        )


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns. Engine
    advances ~90 ticks/turn → reachable max = 93 + 90·(N-1)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        max_turns = pack.levels[lvl].max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True
        )

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        wts: list = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_progression_wins(level, seed):
    """The intended ordered progression must WIN on every
    (level, seed) and complete all five stages."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended progression must WIN on {level} s={seed}; "
        f"got {res.outcome} (then_progress={tp}, "
        f"kills={res.signals.units_killed}, "
        f"buildings={res.signals.own_building_types})"
    )
    assert res.signals.enemy_buildings_destroyed == 0, (
        f"intended policy razed an enemy building on {level} s={seed} "
        f"— it must win on kills only"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed) — the
    chain never advances and the episode times out past within_ticks."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_skip_powr_loses(level, seed):
    """Skipping STAGE 1 (build proc directly) must LOSE — the engine
    refuses proc without power, so the chain never leaves stage 0."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _skip_powr_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"skip-powr must LOSE on {level} s={seed}; "
        f"got {res.outcome} then_progress={tp}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_build_no_attack_loses(level, seed):
    """Building the full economy but never engaging must LOSE —
    STAGE 5 (units_killed_gte:K) never latches; the chain stalls
    at stage 4."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _build_no_attack_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"build-no-attack must LOSE on {level} s={seed}; "
        f"got {res.outcome} then_progress={tp} "
        f"kills={res.signals.units_killed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_hunt_brute_loses(level, seed):
    """A hunt-everything brute that razes the sentinel facts must
    LOSE (a real LOSS via enemy_buildings_destroyed_gte, never a
    DRAW from the engine auto-`done`)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _hunt_brute_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"hunt-brute must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_keep_two_spawn_groups(seed):
    """Hard's two spawn_point groups round-robin by seed — smoke-test
    the spawn-variation contract enforced by test_hard_tier.py."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
