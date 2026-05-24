"""No-cheat + solvency proof for `mcv-deploy-near-resource` (Group A
econ-first siting seed).

The pack tests MCV site choice: deploy the starter MCV at the near ore
patch (so the production base is co-located with its raw-resource
source — Weber facility-location / SC2LE worker travel minimization).
Three plausible-but-wrong policies must LOSE, the intended deploy-in-
place + near-patch-harvest policy must WIN every tier and every hard
seed (1-4):

  - STALL (only `observe`) — 0 income; clock bites → LOSS.
  - DEPLOY-FAR (move MCV to mid/far patch, deploy there) — fact lands
    outside the near-patch region → win predicate never fires → LOSS.
  - NO-DEPLOY (skip the MCV deploy entirely, just harvest) — income
    flows from the pre-placed proc+harv, but the fact-in-region
    predicate is never satisfied → LOSS.
  - HARV-FAR (deploy in place but harvest a distant decoy patch only)
    — fact predicate OK, but harvester cycle to the far patch yields
    too few cr/turn to clear the EV bar → LOSS (medium/hard).

The intended INTENDED policy:
  - turn 1: `Command.deploy([mcv_id])` (creates fact at the spawn,
    inside the near-patch region),
  - every turn: `Command.harvest(harv_ids, near_patch_x, near_patch_y)`.

The fail clause is `after_ticks` ONLY (timeout), aligned with the
within_ticks deadline so a non-finisher LOSES (never draws). The
deploy-location requirement lives ONLY in win_condition — if it were
in fail_condition the stall would insta-LOSS on turn 1 (no fact yet),
hiding the timeout-discrimination signal.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "mcv-deploy-near-resource.yaml"


# ───────────────────────── scripted policies ──────────────────────────────


def _near_patch_for_harv(rs):
    """Hard tier has two spawn groups (y=12 and y=28). Read the
    harvester's actual y and return the matching near-patch coords.
    Easy/medium always y=20."""
    own = rs.get("units_summary", []) or []
    hy = next(
        (u["cell_y"] for u in own if str(u.get("type", "")).lower() == "harv"),
        20,
    )
    if hy < 18:
        return (60, 12)
    if hy > 22:
        return (60, 28)
    return (60, 20)


def _deploy_target(rs):
    """Read MCV row → matching deploy cell ADJACENT to the spawn-
    local ore patch (so the resulting fact lands inside the
    building_in_region predicate's 6-cell radius around the patch
    centre)."""
    own = rs.get("units_summary", []) or []
    my = next(
        (u["cell_y"] for u in own if str(u.get("type", "")).lower() == "mcv"),
        20,
    )
    if my < 18:
        return (61, 12)
    if my > 22:
        return (61, 28)
    return (61, 20)


def stall_policy(rs, Command):
    return [Command.observe()]


def intended_factory():
    """Drive the MCV east to the spawn-local ore patch, deploy adjacent
    to the patch (so the new fact lands inside the building_in_region
    predicate). The pre-placed harv auto-cycles to the spawn-local
    ore patch with no explicit harvest order; issuing one interrupts
    the auto-cycle and stalls income — leave it alone."""
    state = {"moved": False, "deployed": False}

    def policy(rs, Command):
        cmds = []
        own = rs.get("units_summary", []) or []
        mcvs = [u for u in own if str(u.get("type", "")).lower() == "mcv"]
        if mcvs and not state["deployed"]:
            m = mcvs[0]
            dx, dy = _deploy_target(rs)
            if not state["moved"]:
                cmds.append(
                    Command.move_units([str(m["id"])], target_x=dx, target_y=dy)
                )
                state["moved"] = True
            elif abs(m["cell_x"] - dx) <= 2 and abs(m["cell_y"] - dy) <= 2:
                cmds.append(Command.deploy([str(m["id"])]))
                state["deployed"] = True
        return cmds or [Command.observe()]

    return policy


def deploy_far_factory(target_x, target_y):
    """Move MCV to a far-of-spawn cell and deploy THERE. The fact
    lands outside the near-patch region → win predicate fails.
    Harvesting still works via the pre-placed proc, so income may even
    clear the EV bar — but the deploy-location predicate is the
    discriminator."""
    state = {"moved": False, "deployed": False}

    def policy(rs, Command):
        cmds = []
        own = rs.get("units_summary", []) or []
        mcvs = [u for u in own if str(u.get("type", "")).lower() == "mcv"]
        harvs = [
            str(u["id"]) for u in own if str(u.get("type", "")).lower() == "harv"
        ]
        if mcvs and not state["deployed"]:
            m = mcvs[0]
            if not state["moved"]:
                cmds.append(
                    Command.move_units(
                        [str(m["id"])], target_x=target_x, target_y=target_y
                    )
                )
                state["moved"] = True
            elif abs(m["cell_x"] - target_x) <= 4 and abs(m["cell_y"] - target_y) <= 4:
                cmds.append(Command.deploy([str(m["id"])]))
                state["deployed"] = True
        if harvs:
            px, py = _near_patch_for_harv(rs)
            cmds.append(Command.harvest(harvs, px, py))
        return cmds or [Command.observe()]

    return policy


def no_deploy_factory():
    """Skip the deploy entirely — harvester runs but no fact ever
    appears at the near patch → win predicate never fires."""
    def policy(rs, Command):
        own = rs.get("units_summary", []) or []
        harvs = [
            str(u["id"]) for u in own if str(u.get("type", "")).lower() == "harv"
        ]
        if harvs:
            px, py = _near_patch_for_harv(rs)
            return [Command.harvest(harvs, px, py)]
        return [Command.observe()]
    return policy


def harv_far_factory(far_x, far_y):
    """Deploy correctly (fact predicate satisfied) but command the
    harvester to a distant decoy patch — the round-trip cycle is too
    long for income to clear the EV bar (medium/hard)."""
    state = {"deployed": False}

    def policy(rs, Command):
        cmds = []
        own = rs.get("units_summary", []) or []
        mcvs = [
            str(u["id"]) for u in own if str(u.get("type", "")).lower() == "mcv"
        ]
        harvs = [
            str(u["id"]) for u in own if str(u.get("type", "")).lower() == "harv"
        ]
        if mcvs and not state["deployed"]:
            cmds.append(Command.deploy(mcvs))
            state["deployed"] = True
        if harvs:
            cmds.append(Command.harvest(harvs, far_x, far_y))
        return cmds or [Command.observe()]

    return policy


# ───────────────────────── helpers ────────────────────────────────────────


def _ev(res):
    return res.signals.cash + res.signals.resources


def _run(level, policy_factory, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    return c, run_level(c, policy_factory(), seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_and_meta():
    pack = load_pack(PACK)
    assert pack.meta.status == "active"
    assert pack.meta.id == "mcv-deploy-near-resource"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor
    assert any("SC2LE" in a for a in anchors), anchors
    assert any("MicroRTS" in a for a in anchors), anchors
    assert any("Weber" in a for a in anchors), anchors


def test_hard_has_two_spawn_point_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"


def test_all_tiers_have_reachable_deadlines():
    """within_ticks and after_ticks must sit at-or-below the engine
    ceiling 93 + 90·(max_turns - 1) so the clock actually bites."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        ceiling = 93 + 90 * (L.max_turns - 1)
        win_clauses = L.win_condition.model_dump().get("all_of", [])
        wt = next(int(c["within_ticks"]) for c in win_clauses if "within_ticks" in c)
        ft = next(
            int(c["after_ticks"])
            for c in L.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        # after_ticks one tick after within_ticks: non-finisher LOSES.
        assert ft == wt + 1, f"{lvl}: after_ticks {ft} should == within_ticks+1 ({wt+1})"


# ───────────────────────── intended WINS (seeds 1-4 on hard) ──────────────


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_easy_intended_wins(seed):
    _, r = _run("easy", intended_factory, seed=seed)
    assert r.outcome == "win", (
        f"easy intended seed={seed} should WIN, got {r.outcome} "
        f"ev={_ev(r)} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_medium_intended_wins(seed):
    _, r = _run("medium", intended_factory, seed=seed)
    assert r.outcome == "win", (
        f"medium intended seed={seed} should WIN, got {r.outcome} "
        f"ev={_ev(r)} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_intended_wins(seed):
    _, r = _run("hard", intended_factory, seed=seed)
    assert r.outcome == "win", (
        f"hard intended seed={seed} should WIN, got {r.outcome} "
        f"ev={_ev(r)} tick={r.signals.game_tick}"
    )


# ───────────────────────── no-cheat: lazy / wrong plays LOSE ──────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_on_every_level(level, seed):
    """Stall = 0 income (starting_cash:0) AND no deploy → both win
    clauses fail → clock bites → reachable timeout LOSS (not a draw)."""
    _, r = _run(level, lambda: stall_policy, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed} stall must be a real timeout LOSS, got "
        f"{r.outcome} (tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_deploy_far_loses_easy(seed):
    """Move MCV to a far patch (50,18 mid-map) and deploy there: fact
    lands outside the (12,18)±5 near-patch region → win never fires.
    Income may still flow from the pre-placed proc+harv, but the
    deploy-location predicate is the discriminator."""
    _, r = _run("easy", lambda: deploy_far_factory(50, 18), seed=seed)
    assert r.outcome == "loss", (
        f"easy/seed{seed} deploy-far must LOSE; got {r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_deploy_far_loses_medium(seed):
    _, r = _run("medium", lambda: deploy_far_factory(50, 18), seed=seed)
    assert r.outcome == "loss", (
        f"medium/seed{seed} deploy-far must LOSE; got {r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_deploy_far_loses_hard(seed):
    """Hard tier: deploy at the shared FAR decoy patch (100,20). Fact
    is far from EITHER spawn-local near region → win never fires on
    any seed."""
    _, r = _run("hard", lambda: deploy_far_factory(100, 20), seed=seed)
    assert r.outcome == "loss", (
        f"hard/seed{seed} deploy-far must LOSE; got {r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_no_deploy_loses(level, seed):
    """Harvest only, never deploy the MCV: income flows from the
    pre-placed proc+harv but no fact ever appears in the near-patch
    region → win predicate never fires → clock bites → LOSS."""
    _, r = _run(level, no_deploy_factory, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed} no-deploy must LOSE; got {r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_harv_far_loses_medium(seed):
    """Medium: deploy in place (fact predicate OK) but tell the
    harvester to harvest the FAR decoy at (50,18). The proc is fixed
    at (10,18) → round-trip is long → ~33 cr/turn yields only ~1500
    ev by the deadline, below the 4500 bar."""
    _, r = _run("medium", lambda: harv_far_factory(50, 18), seed=seed)
    assert r.outcome == "loss", (
        f"medium/seed{seed} harv-far must LOSE on income gate; got "
        f"{r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_harv_far_loses_hard(seed):
    """Hard: deploy in place but harvest the shared FAR decoy at
    (100,20). Income tops out near baseline → below the 5500 bar."""
    _, r = _run("hard", lambda: harv_far_factory(100, 20), seed=seed)
    assert r.outcome == "loss", (
        f"hard/seed{seed} harv-far must LOSE on income gate; got "
        f"{r.outcome} ev={_ev(r)}"
    )


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared
    spawn_point groups (north vs south) so a memorised opening can't
    generalise."""
    from pathlib import Path

    from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter

    c = compile_level(load_pack(PACK), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts = set()
    try:
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            if u:
                starts.add(
                    tuple(sorted((x["cell_x"], x["cell_y"]) for x in u))
                )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds 1-4 produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, intended_factory(), seed=2)
    b = run_level(c, intended_factory(), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
