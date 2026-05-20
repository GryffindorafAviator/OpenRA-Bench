"""def-surprise-flank-react scenario family, full loop on Rust.

The pack tests SURPRISE-AXIS REACTION: the brief states the enemy will
attack from the NORTH and the world is pre-rigged accordingly (2
pillboxes + 4 defenders pre-positioned on the NORTH lane). The actual
rusher band arrives from the SOUTH and walks around the entire NORTH
defence line. The intended capability is detecting the actual attack
axis from the observation and RE-POSITIONING the four pre-placed
defenders to intercept the real rush. The pre-built NORTH pillboxes
are dead wood for this engagement; the capability is the reactive
re-positioning, not the build.

The win predicate makes all three axes load-bearing:

* `has_building:fact` ⇒ the fact must survive (stall / stay-NORTH /
  pure-build all let it get razed by the south rush);
* `units_killed_gte:4` ⇒ the south rush must actually be engaged
  (stay-NORTH never engages, the 4 defenders sit at y≈14 while the
  rusher band walks past them at y≈20..32);
* `own_units_gte:3` ⇒ ≥3 defenders must survive (the south rush
  reaching the fact splashes the close defenders even if some
  trade);
* `within_ticks:4500` paired with `after_ticks:4501` ⇒ a non-finisher
  is a real reachable timeout LOSS (50 turns × ≤90 ticks/step reaches
  ≥4503 in interrupt mode), never a draw.

These tests prove with deterministic scripted policies (no model, no
network) that:

* the intended detect-and-redeploy policy WINS every level + every
  hard seed (1..4);
* stall / stay-NORTH / pure-build all LOSE every level + every hard
  seed (a real LOSS, not a draw);
* the `after_ticks` deadline is reachable inside `max_turns`;
* the hard tier defines ≥2 spawn_point groups (so the base latitude
  and matching surprise-axis vary by seed — anti-memorisation).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-surprise-flank-react.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ─────────────────────────────────────────────────


def _fact_xy(rs):
    fact = next(
        (b for b in (rs.get("own_buildings") or [])
         if b.get("type") == "fact"),
        None,
    )
    if fact is None:
        return None
    return int(fact["cell_x"]), int(fact["cell_y"])


def _enemies(rs):
    return [
        e for e in (rs.get("enemy_positions") or [])
    ]


def stall(rs, C):
    """Observe-only — defenders never move; the south rush walks
    past the pre-built NORTH defence and razes the fact."""
    return [C.observe()]


def stay_north(rs, C):
    """Stay-NORTH: explicitly issue an attack_move toward the NORTH
    pillbox line (where the intel said the rush would arrive). The
    defenders re-park on the WRONG axis and never engage the actual
    south rusher band; fact razed → LOSS.
    """
    units = rs.get("units_summary") or []
    own = [u for u in units if str(u.get("type", "")).lower() in ("e1", "e3")]
    if not own:
        return [C.observe()]
    xy = _fact_xy(rs)
    if xy is None:
        return [C.observe()]
    fx, fy = xy
    # Park defenders NORTH of the fact (y_fact - 6). This is where
    # the pre-built pillbox line sits (the intel direction).
    target_y = max(2, fy - 6)
    ids = [str(u["id"]) for u in own]
    return [C.attack_move(ids, target_x=fx, target_y=target_y)]


def pure_build(rs, C):
    """Pure-build: try to spend the starting cash on an extra building
    near the existing NORTH defence (or anywhere) but NEVER re-position
    the defenders. With cash 800 and no tent the infantry queue is
    empty; the model can only queue a pbox (600cr) or a powr. Either
    way the south rush reaches the fact unopposed → LOSS.
    """
    own_b = rs.get("own_buildings") or []
    prod = rs.get("production") or []
    prod_items = [
        p.get("item") for p in prod if isinstance(p, dict)
    ]
    xy = _fact_xy(rs)
    if xy is None:
        return [C.observe()]
    fx, fy = xy
    n_pbox = sum(1 for b in own_b if b.get("type") == "pbox")
    cmds = []
    # Try to queue + place a third NORTH pbox (the "intel" lane).
    if n_pbox < 3:
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        cmds.append(C.place_building("pbox", fx + 4, fy - 6))
    if not cmds:
        cmds.append(C.observe())
    return cmds


def make_intended():
    """Detect-and-redeploy: observe the spotted rusher units (which
    arrive on the SURPRISE axis, not the intel axis), then commit all
    four defenders to engage them. The 4 defenders intercept the rush
    before/at the fact, focus-fire on the visible rusher cluster,
    kill ≥4 and keep ≥3 alive; fact survives → WIN.
    """

    def policy(rs, C):
        units = rs.get("units_summary") or []
        own = [u for u in units if str(u.get("type", "")).lower() in ("e1", "e3")]
        if not own:
            return [C.observe()]
        xy = _fact_xy(rs)
        if xy is None:
            return [C.observe()]
        fx, fy = xy
        ids = [str(u["id"]) for u in own]
        # Detect actual rush axis from any visible enemy (rs lists
        # them in `enemy_summary`; the fogged `enemy_positions` is
        # empty for this pack's start). If any rusher is visible,
        # commit ALL defenders onto it (attack_unit on the nearest
        # rusher → focus-fire). Otherwise march toward the OPPOSITE
        # side of the base from the pre-built pbox line — the
        # surprise axis the brief warns about.
        es = rs.get("enemy_summary") or []
        # Filter out the inert far-east fact marker (and any other
        # buildings); only mobile combat units.
        rusher_units = [
            e for e in es
            if str(e.get("type", "")).lower() in ("e1", "e3")
        ]
        if rusher_units:
            # Nearest rusher to the fact (the immediate threat).
            tgt = min(
                rusher_units,
                key=lambda e: (
                    (int(e.get("cell_x", fx)) - fx) ** 2
                    + (int(e.get("cell_y", fy)) - fy) ** 2
                ),
            )
            eid = str(tgt.get("id"))
            return [C.attack_unit(ids, eid)]
        # No rusher visible yet — march toward the surprise axis
        # (opposite the pre-built pbox line).
        own_b = rs.get("own_buildings") or []
        pboxes = [b for b in own_b if b.get("type") == "pbox"]
        if pboxes:
            pby = sum(int(b["cell_y"]) for b in pboxes) // len(pboxes)
            ty = fy + 6 if pby < fy else fy - 6
            tx = fx
        else:
            return [C.observe()]
        return [C.attack_move(ids, target_x=tx, target_y=ty)]

    return policy


# ── scenario-shape invariants ─────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-surprise-flank-react"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    anchors = pack.meta.benchmark_anchor
    assert any("adversarial robustness" in a for a in anchors), anchors
    assert any("CICERO" in a for a in anchors), anchors
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert (str(bot).lower() == "rusher"), (lvl, bot)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail must be
    strictly below the tick reachable at max_turns (≤90 ticks/step in
    interrupt mode)."""
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
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the base latitude (and matching surprise axis) varies by seed."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126):
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_pre_built_north_defense_present_easy_and_medium():
    """The PRE-BUILT NORTH pillbox line + four pre-placed NORTH
    defenders must be present on easy and medium (the "trap" the
    intended play must walk away from)."""
    for level in ("easy", "medium"):
        c = compile_level(load_pack(PACK), level)
        pboxes = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "pbox"
        ]
        assert len(pboxes) == 2, (level, pboxes)
        defenders = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "e1"
        ]
        assert len(defenders) == 4, (level, defenders)


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_detect_and_redeploy_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended detect-and-redeploy play "
            f"must WIN; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every lazy / wrong-axis policy LOSES (not draws) ───────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall", lambda: stall),
        ("stay_north", lambda: stay_north),
        ("pure_build", lambda: pure_build),
    ],
)
def test_lazy_and_wrong_axis_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (fact razed), stay-NORTH (defenders park on the intel
    lane while the south rush walks past), and pure-build (more
    buildings on the intel lane, defenders never re-positioned) must
    ALL LOSE on every level + every seed — no draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy_factory(), seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── determinism ───────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended(), seed=3)
    b = run_level(c, make_intended(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
