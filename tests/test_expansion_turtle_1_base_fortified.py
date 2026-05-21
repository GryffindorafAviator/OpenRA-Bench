"""expansion-turtle-1-base-fortified — Wave-4 Group B TURTLE 1-base
fortification.

Scripted-policy no-defect / no-cheat validation. The pack hands the
agent a SINGLE pre-placed base (fact + tent + 2× powr + proc + harv +
mine + a SEED defence line of pre-placed pbox + 1 gun + weap) PLUS a
small starting screen, and asks the agent to FORTIFY it densely
enough to repel an incoming hunt-bot assault while reaching a higher
defence bar (≥3/≥6/≥8 pbox + ≥1/≥2/≥2 gun) and maintaining ≥3
defenders alive, all before tick 5400 with the conyard still standing.

Expansion is structurally forbidden: no MCV is provided and `deploy`
is removed from the tools list. The discrimination is whether the
model embraces the deep-defence play (actively REINFORCES the
pre-placed pbox / gun seed against the hunt's grinding attrition)
instead of stalling, building pure army (no pbox replacement, defences
bleed below the bar), or over-teching (cash spent on dome/fix
research instead of pbox/gun reinforcement).

The intended bar (CLAUDE.md): every lazy / single-axis / wrong-axis
policy LOSES on every level and every hard seed (1–4); only the
defence-reinforcement policy WINS.

  - stall                     (only Command.observe())             → LOSS
  - pure_army                 (only train e1, no defence builds)    → LOSS
  - over_tech                 (build dome+fix instead of pbox/gun)  → LOSS
  - intended_fortify          (reinforce pbox+gun, train e1)        → WIN
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "expansion-turtle-1-base-fortified.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-level (pbox_target, gun_target) — must match the pack YAML.
PARAMS = {"easy": (3, 1), "medium": (6, 2), "hard": (8, 2)}


# ── helpers ───────────────────────────────────────────────────────────


def _norm_prod(rs):
    """Flatten the production list to item names (the engine emits
    either dicts or bare strings depending on queue/version)."""
    out = []
    for p in (rs.get("production") or []):
        if isinstance(p, dict):
            out.append(p.get("item"))
        elif isinstance(p, str):
            out.append(p)
    return out


def _fact_xy(rs):
    fact = next(
        (b for b in (rs.get("own_buildings") or []) if b.get("type") == "fact"),
        None,
    )
    if fact is None:
        return None
    return int(fact["cell_x"]), int(fact["cell_y"])


def _building_counts(rs):
    own = rs.get("own_buildings") or []
    counts = {}
    for b in own:
        counts[b.get("type")] = counts.get(b.get("type"), 0) + 1
    return counts


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — pre-placed pbox / gun get bled down by the
    hunt band; defences fall below the bar, screen wipes below 3."""
    return [C.observe()]


def pure_army(rs, C):
    """Only ever train e1, never reinforce defences. Pre-placed pbox
    bleeds below the bar; bar unmet at the deadline."""
    prod = _norm_prod(rs)
    if "e1" not in prod:
        return [C.build("e1")]
    return [C.observe()]


def over_tech(rs, C):
    """Build research buildings (dome, fix) instead of defence
    reinforcement. Cash spent on tech; pbox bar bleeds below target."""
    own = rs.get("own_buildings") or []
    types = {b.get("type") for b in own}
    prod = _norm_prod(rs)
    xy = _fact_xy(rs)
    if xy is None:
        return [C.observe()]
    fx, fy = xy
    cmds = []
    if "dome" not in types and "dome" not in prod:
        cmds.append(C.build("dome"))
    if "dome" not in types:
        cmds.append(C.place_building("dome", fx - 6, fy + 6))
    elif "fix" not in types and "fix" not in prod:
        cmds.append(C.build("fix"))
        cmds.append(C.place_building("fix", fx - 8, fy + 6))
    if not cmds:
        cmds.append(C.observe())
    return cmds


def make_intended(pbox_target: int, gun_target: int):
    """Intended fortification policy: continuously REINFORCE the
    pre-placed pbox seed against the hunt band's grinding attrition,
    secure the gun bar once pbox is above a survival floor, and train
    a stream of replacement infantry to maintain own_units_gte:3 as
    the starting screen takes damage.

    Defence queue is SERIAL (Defense), Infantry queue is PARALLEL.
    The Defence queue is held continuously full: the moment it is
    empty the policy queues the next item. Priority order:
      1. if pbox is below the survival floor (target-2) → build pbox;
      2. else if a gun is still needed (and weap is online) → gun;
      3. else top pbox up to target+5 so attrition is always covered.

    Placement spams the closest free cells each turn (the engine
    takes the first valid one); a cell freed by a destroyed pbox is
    re-used, so the defensive wall is rebuilt in place rather than
    marching ever-eastward into the open.
    """

    def policy(rs, C):
        own = rs.get("own_buildings") or []
        occupied = {(b["cell_x"], b["cell_y"]) for b in own}
        # Treat unit-occupied cells as blocked too — a pbox cannot
        # place onto a cell a unit is standing on, and a blocked
        # placement silently stalls the serial Defence queue.
        for u in (rs.get("units_summary") or []):
            if u.get("cell_x") is not None:
                occupied.add((u["cell_x"], u["cell_y"]))
        pbox_count = sum(1 for b in own if b.get("type") == "pbox")
        gun_count = sum(1 for b in own if b.get("type") == "gun")
        weap_count = sum(1 for b in own if b.get("type") == "weap")
        prod = _norm_prod(rs)
        n_pbox = prod.count("pbox")
        n_gun = prod.count("gun")
        n_e1 = prod.count("e1")
        xy = _fact_xy(rs)
        if xy is None:
            return [C.observe()]
        fx, fy = xy
        units = rs.get("units_summary") or []
        n_units = sum(
            1
            for u in units
            if str(u.get("type", "")).lower() in ("e1", "e3")
        )
        cmds = []
        # Candidate placement cells east of fact, in bounds
        # (y∈[2,38], x∈[2,126]), nearest-to-fact first.
        cells = []
        for dx in range(3, 24):
            for dy in range(-10, 11):
                cx, cy = fx + dx, fy + dy
                if 2 <= cx <= 126 and 2 <= cy <= 38:
                    cells.append((cx, cy))
        cells.sort(key=lambda c: abs(c[0] - fx) + abs(c[1] - fy))
        free = [c for c in cells if c not in occupied]

        # Defence queue (Serial): keep it continuously full.
        if n_pbox + n_gun == 0:
            need_gun = gun_count < gun_target and weap_count >= 1
            if pbox_count < pbox_target - 2:
                cmds.append(C.build("pbox"))
            elif need_gun:
                cmds.append(C.build("gun"))
            elif pbox_count < pbox_target + 5:
                cmds.append(C.build("pbox"))

        # Place in-flight defence — spam the closest free cells so a
        # freed (destroyed-pbox) cell is reclaimed and the queue is
        # never stalled by an invalid target cell.
        if n_pbox >= 1:
            for cx, cy in free[:6]:
                cmds.append(C.place_building("pbox", cx, cy))
        if n_gun >= 1:
            for cx, cy in free[:6]:
                cmds.append(C.place_building("gun", cx, cy))

        # Infantry queue (Parallel): keep training replacements so
        # own_units_gte:3 holds through screen attrition.
        if n_units < 6 and n_e1 == 0:
            cmds.append(C.build("e1"))

        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


# ── scenario-shape invariants ─────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_hunt_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "expansion-turtle-1-base-fortified"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    anchors = pack.meta.benchmark_anchor
    assert any("fortress" in a.lower() or "1-base" in a for a in anchors), anchors
    assert any("market" in a.lower() or "deeply" in a for a in anchors), anchors
    assert any("fortress doctrine" in a.lower() or "impregnable" in a.lower() for a in anchors), anchors
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "hunt", (lvl, bot)


def test_deploy_not_in_tools():
    """TURTLE contract: `deploy` is structurally absent so the agent
    cannot expand (the bench advertises only what the agent can do)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        tools = [str(t).lower() for t in c.scenario.tools]
        assert "deploy" not in tools, f"{lvl}: deploy must not be in tools"


def test_no_mcv_provided():
    """Belt-and-suspenders TURTLE contract: no MCV pre-placed either,
    so even if the deploy tool sneaks back the agent cannot deploy
    what does not exist."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        mcvs = [a for a in c.scenario.actors if a.type == "mcv"]
        assert mcvs == [], f"{lvl}: no MCV must be pre-placed; got {mcvs}"


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail must be
    strictly below the tick reachable at max_turns."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    deadline = None
    for clause in fc.get("any_of", []) or []:
        if "after_ticks" in clause:
            deadline = int(clause["after_ticks"])
    assert deadline is not None, f"{level}: no after_ticks fail clause"
    reachable = 93 + 90 * (c.max_turns - 1)
    assert deadline < reachable, (
        f"{level}: deadline {deadline} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → DRAW degeneracy"
    )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the home base latitude flips per seed (anti-memorisation)."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # All actors must be inside the rush-hour-arena playable bounds.
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_win_uses_per_tick_fact_count_not_monotonic_has_building():
    """CLAUDE.md footgun: `has_building` is monotonic (never un-sets
    after the fact dies), so the "conyard still standing" clause
    must use `building_count_gte:{type:fact,n:1}` (per-tick
    refreshed) — otherwise an in-progress razed-conyard run still
    "passes" the predicate."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        clauses = win.get("all_of", []) or []
        fact_keep = any(
            isinstance(cl.get("building_count_gte"), dict)
            and cl["building_count_gte"].get("type") == "fact"
            for cl in clauses
        )
        assert fact_keep, (
            f"{lvl}: 'fact still standing' clause must use "
            f"building_count_gte (per-tick refreshed), not has_building "
            f"(monotonic)."
        )


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_fortify_policy_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    pbox_t, gun_t = PARAMS[level]
    for seed in SEEDS:
        r = run_level(c, make_intended(pbox_t, gun_t), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended dense-fortification "
            f"reinforcement policy must WIN; got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every lazy / single-axis policy LOSES (not draws) ──────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy",
    [
        ("stall", stall),
        ("pure_army", pure_army),
        ("over_tech", over_tech),
    ],
)
def test_lazy_and_single_axis_policies_lose_every_level_and_seed(
    level, policy_name, policy
):
    """Stall (defences bleed below bar), pure-army (no defence
    replacement → pbox falls below bar), over-tech (cash spent on
    dome/fix → no pbox reinforcement, bar unmet) must ALL LOSE on
    every level + every seed — a real reachable LOSS, never a draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── determinism ───────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    pbox_t, gun_t = PARAMS["easy"]
    a = run_level(c, make_intended(pbox_t, gun_t), seed=3)
    b = run_level(c, make_intended(pbox_t, gun_t), seed=3)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
