"""Builds audits/family10_special.csv — Family-10 (Special weapons + Misc) manual audit.

One row per (pack, level). Each briefing is SELF-CONTAINED in F1 officer
style — the model sees one level at a time, so every briefing fully
describes the mission framing, the forces given (with positions where
they matter), and the objective from scratch. No "same as before" or
"the same X" references.

Structure per briefing: mission framing → what is given → target/objective.
Red-Alert-specific terms (spy, thief, engineer, Tanya, mslo, etc.) are
explained inline so non-RA readers can follow.

The map_fit column flags scenarios where the map is too large for the
actual decision under test. Most F10 spec-* packs ship a tailored arena
already (one map per tier, sized to the specific cell); rush-hour stays
on 128x40 by design (its capability IS the whole-arena sweep);
custom-map-no-enemy uses the bespoke singles-maginot custom map.

Scope (7 packs × 3 levels = 21 rows):
- 5 spec-* packs (engineer / nuke / spy / Tanya / thief)
- rush-hour (historical baseline — DO NOT edit unless explicitly invoked)
- custom-map-no-enemy (pure navigation, no enemy)

TEMPLATE.yaml is scaffolding and is NOT included.
"""
import csv
from pathlib import Path

OUT = Path(__file__).parent / 'family10_special.csv'
R = []


def add(pack, level, cap, map_name, map_size, map_fit, tools, agent, enemy,
        briefing, win, lose, max_turns, tick_budget,
        posture='', posture_issue=''):
    R.append(dict(
        pack=pack, level=level, capability=cap, map_name=map_name,
        map_size=map_size, map_fit=map_fit, tools=tools,
        agent_force=agent, enemy_force=enemy,
        briefing_RA=briefing, win_condition=win, lose_condition=lose,
        max_turns=max_turns, tick_budget=tick_budget,
        enemy_posture=posture, posture_issue=posture_issue,
    ))


# ── 1. spec-engineer-capture — capture an enemy proc with `e6` engineer
P = 'spec-engineer-capture'; C = 'action'
T = 'observe, move_units, attack_unit, capture_actor'
add(P, 'easy', C, 'engineer-capture-easy-arena', '80x36', 'fit', T,
    '1× engineer (unarmed capture specialist) @(16,18)',
    '1× Ore Refinery (enemy proc) @(20,18); undefended',
    "Commander, this is a single-asset seizure. You are given one engineer — an unarmed specialist whose only useful action is to walk into an enemy structure and transfer its ownership to you, consuming the engineer in the process. An enemy Ore Refinery sits four cells east of your engineer on an open arena. The refinery has no defenders. Seize it within about 25 turns; the engineer carries no weapon and an attack order on the building is a no-op.",
    "Agent owns a proc (i.e. the enemy refinery is captured), within 2000 ticks.",
    "Deadline (2001 ticks).",
    25, 2253,
    'none (undefended target)', '')
add(P, 'medium', C, 'engineer-capture-chokepoint-arena', '112x40', 'fit', T,
    '1× engineer @(12,20) in the west lobe',
    '1× Ore Refinery @(96,20) in the east lobe; undefended',
    "Commander, this is a single-asset seizure across a chokepoint map. A vertical water wall splits the arena at x=52..60 with a single 4-cell corridor at y=18..21 — that corridor is the only route from west to east. You are given one engineer — an unarmed specialist whose only useful action is to walk into an enemy structure and transfer its ownership to you, consuming the engineer. The target Ore Refinery sits deep in the east lobe at (96,20); pathing finds the corridor automatically. Seize the refinery within about 35 turns.",
    "Agent owns a proc, within 3000 ticks.",
    "Deadline (3001 ticks).",
    35, 3153,
    'none (undefended target; pathing test)', '')
add(P, 'hard', C, 'engineer-capture-serpentine-arena', '140x44', 'fit', T,
    '2× engineer @(10,22)/(10,23) (redundancy)',
    '1× Ore Refinery @(128,20) far east; 1× enemy rifleman @(128,35) stance:2 (Defend) south of the refinery',
    "Commander, this is a single-asset seizure across a serpentine map. Two staggered water-obstacle clusters force a snaking route across the middle band: the engineer must slip below the first cluster (upper-west) and above the second (lower-east) to reach an Ore Refinery on the far east at (128,20). One enemy rifleman stands south of the refinery on Defend stance — he covers a south approach but cannot range a unit arriving from the north. You are given two engineers for redundancy — unarmed specialists whose only useful action is to walk into an enemy structure and transfer its ownership to you, consuming the engineer. Seize the refinery within about 50 turns; the clock is tight.",
    "Agent owns a proc, within 4000 ticks.",
    "Deadline (4001 ticks).",
    50, 4503,
    'static (st2 e1 defender south of objective)',
    'NOTE: defender at (128,35) cannot reach engineer arriving from the north — geometry forces the safe approach. Intended.')


# ── 2. spec-nuke-strike — fire `mslo` nuke at an enemy silo cluster
P = 'spec-nuke-strike'; C = 'action'
T = 'observe, fire_superweapon'
add(P, 'easy', C, 'spec-nuke-strike-arena 160x60', '160x60', 'fit', T,
    '1× Nuclear Missile Silo (mslo) @(10,10) NW; 1× jeep spotter @(80,24) (HOLD-FIRE; cannot move or attack)',
    '5× ore silo cluster @(78-82,30); 1× far enemy Construction Yard @(80,38) (engine keep-alive)',
    "Commander, this is a one-shot superweapon strike. You command a Nuclear Missile Silo at the north-west corner — the silo's missile finishes arming about 100 ticks into the mission. Far across the map sits a tight row of five enemy ore silos, and a friendly jeep already has eyes on them. The jeep cannot move and cannot attack — it is a spotter only. Read your observation, take the centroid of the silo cluster, and fire one missile at that cell. Destroy all five silos within about 22 turns.",
    "≥5 enemy buildings destroyed, within 1800 ticks.",
    "Deadline (1801 ticks).",
    22, 1983,
    'static (HoldFire spotter; enemy silos passive)',
    'INTENTIONAL: enemy is passive scenery (the test is the strike, not combat).')
add(P, 'medium', C, 'spec-nuke-strike-arena 160x60', '160x60', 'fit', T,
    '1× Nuclear Missile Silo @(10,10) NW; 1× jeep spotter @(130,49) (HOLD-FIRE)',
    '6× ore silo cluster (5-in-a-row + 1) @(128-132,43)+(130,44); 1× far enemy Construction Yard @(130,51)',
    "Commander, this is a one-shot superweapon strike. You command a Nuclear Missile Silo at the north-west corner; the missile finishes arming about 100 ticks in. A dense cluster of six enemy ore silos sits in the south-east corner of the map and a friendly jeep is in eyeshot — the jeep cannot move and cannot attack. Read your observation, find the cluster centroid, fire one missile. All six silos in one shot, within about 22 turns.",
    "≥6 enemy buildings destroyed, within 1800 ticks.",
    "Deadline (1801 ticks).",
    22, 1983,
    'static (HoldFire spotter)', '')
add(P, 'hard', C, 'spec-nuke-strike-arena 160x60', '160x60', 'fit', T,
    '1× Nuclear Missile Silo @(10,10) (fixed across seeds); 1× jeep spotter @(140,17) sp0 NE or @(20,43) sp1 SW (HOLD-FIRE)',
    '6× ore silo cluster — NE @(138-142,12)+(140,11) sp0 OR SW @(18-22,48)+(20,49) sp1; far enemy Construction Yard keep-alive per seed',
    "Commander, this is a one-shot superweapon strike with seed-rotated target placement. You command a Nuclear Missile Silo at the north-west corner; the missile finishes arming about 100 ticks in. A dense cluster of six enemy ore silos spawns in EITHER the north-east corner OR the south-west corner of the map depending on seed — read your observation to find which. A friendly jeep ships in with the cluster as a spotter (it cannot move or attack). Find the centroid, fire, take out all six silos in one shot, within about 22 turns.",
    "≥6 enemy buildings destroyed, within 1800 ticks.",
    "Deadline (1801 ticks).",
    22, 1983,
    'static (HoldFire spotter)', '')


# ── 3. spec-spy-infiltrate — spy reveal-scan of enemy network
P = 'spec-spy-infiltrate'; C = 'action'
T = 'observe, move_units, attack_unit, infiltrate'
add(P, 'easy', C, 'spy-infiltrate-easy', '112x40', 'fit', T,
    '1× spy (unarmed infiltrator) @(10,20) HOLD-FIRE',
    '1× Ore Refinery @(14,20) close + 1× Power Plant @(100,6) + 1× Barracks @(105,30) + 1× Ore Silo @(108,18) — last three hidden in fog',
    "Commander, this is an intelligence-by-infiltration mission. You are given one spy — an unarmed infiltrator whose only useful action is to walk into an enemy building and trigger a one-shot intel scan: every structure owned by that operator is added to your map, surviving fog of war. The spy is consumed by the scan. An enemy Ore Refinery sits four cells east of your spy; two cover clusters flank the approach (north and south). Three more enemy structures are hidden in the eastern half of the map. Reveal at least four enemy structures within about 25 turns. The spy carries no weapon; an attack order is a no-op.",
    "≥4 enemy buildings discovered, within 2000 ticks.",
    "Deadline (2001 ticks), or spy destroyed.",
    25, 2253,
    'none (no defenders)', '')
add(P, 'medium', C, 'spy-infiltrate-medium', '112x40', 'fit', T,
    '1× spy @(10,20) HOLD-FIRE',
    '1× Ore Refinery @(15,20) + 5 hidden enemy structures along east half (Power Plants, Barracks, Ore Silo, Construction Yard)',
    "Commander, this is an intelligence-by-infiltration mission with route choice. You are given one spy — an unarmed infiltrator whose only useful action is to walk into an enemy building and trigger a one-shot intel scan: every structure owned by that operator is added to your map, surviving fog. The spy is consumed by the scan. An enemy Ore Refinery sits five cells east of your spy; cover clusters flank the central lane to north and south. Five more enemy structures sit hidden in the eastern half of the map. Reveal at least six enemy structures within about 35 turns.",
    "≥6 enemy buildings discovered, within 3000 ticks.",
    "Deadline (3001 ticks), or spy destroyed.",
    35, 3153,
    'none (no defenders)', '')
add(P, 'hard', C, 'spy-infiltrate-hard', '128x40', 'fit', T,
    '2× spy @(11,18)/(11,22) HOLD-FIRE (redundancy)',
    '1× Ore Refinery @(15,20) + 1× enemy rifleman @(18,32) stance:2 (Defend) covering south detour + 6 hidden enemy structures across east half',
    "Commander, this is an intelligence-by-infiltration mission against a defended target. You are given two spies (redundancy) — unarmed infiltrators whose only useful action is to walk into an enemy building and trigger a one-shot intel scan: every structure owned by that operator is added to your map, surviving fog. Each spy is consumed by its scan. An enemy Ore Refinery sits five cells east, with cover clusters to north and south of the approach lane. One enemy rifleman is dug in well south of the refinery on Defend stance — his rifle ranges any wanderer that drifts south, but a direct walk along the central corridor stays outside his range. Six more enemy structures sit hidden in the eastern half. Reveal at least seven enemy structures within about 50 turns.",
    "≥7 enemy buildings discovered, within 4000 ticks.",
    "Deadline (4001 ticks), or no spy left.",
    50, 4503,
    'static (st2 e1 covering south detour)',
    'NOTE: defender off the direct lane — punishes wander, not the intended beeline. Intended.')


# ── 4. spec-tanya-c4-strike — Tanya C4 instant-destroy of enemy proc
P = 'spec-tanya-c4-strike'; C = 'action'
T = 'observe, move_units, attack_unit, set_stance, c4_detonate'
add(P, 'easy', C, 'spec-tanya-c4-strike-arena 80x32', '80x32', 'fit', T,
    '1× Tanya (elite commando, fast, anti-infantry pistol; C4 instantly demolishes a building on adjacency) @(16,16)',
    '1× Ore Refinery @(22,16); 1× far enemy Construction Yard @(74,16) (engine keep-alive)',
    "Commander, this is a demolition strike. You are given Tanya — an elite commando carrying C4 demolition charges. On adjacency to any enemy building she can plant the charges and instantly destroy it, surviving the blast. The pistol she also carries is anti-infantry only; it cannot demolish a building inside any sensible deadline. An undefended enemy Ore Refinery sits six cells east of her current position on a small open arena. Demolish it within about 20 turns.",
    "≥1 enemy building destroyed, within 1500 ticks.",
    "Deadline (1501 ticks).",
    20, 1803,
    'none (undefended)', '')
add(P, 'medium', C, 'spec-tanya-c4-strike-arena 80x32', '80x32', 'fit', T,
    '1× Tanya @(14,16)',
    '1× Ore Refinery @(22,16) + 2× enemy rifleman @(20,15)/(20,17) stance:0 (HOLD-FIRE — will NOT engage, even if attacked)',
    "Commander, this is a demolition strike past idle defenders. You are given Tanya — an elite commando carrying C4 demolition charges. On adjacency to any enemy building she can plant the charges and instantly destroy it, surviving the blast. The pistol she also carries is anti-infantry only; it cannot demolish a building inside any sensible deadline. An enemy Ore Refinery sits eight cells east; two enemy riflemen flank the refinery on HOLD-FIRE orders — they will NOT open fire even if shot at. The riflemen are bait. Demolish the refinery within about 25 turns; do not get distracted.",
    "≥1 enemy building destroyed, within 2000 ticks.",
    "Deadline (2001 ticks).",
    25, 2253,
    'passive (st0 HoldFire — INTENTIONAL bait)',
    'INTENTIONAL: riflemen will die silently if shot; the test is to ignore them and C4 the refinery.')
add(P, 'hard', C, 'spec-tanya-c4-strike-arena 80x32', '80x32', 'fit', T,
    '1× Tanya @(10,18)',
    '1× Ore Refinery @(24,14) + 2× enemy rifleman @(20,14)/(22,14) stance:2 (Defend — will fire on Tanya in range)',
    "Commander, this is a demolition strike through active cover. You are given Tanya — an elite commando carrying C4 demolition charges. On adjacency to any enemy building she can plant the charges and instantly destroy it, surviving the blast. The pistol she also carries is anti-infantry only; it cannot demolish a building inside any sensible deadline. An enemy Ore Refinery sits about 12 cells away, with two enemy riflemen on Defend stance covering the approach lane — they will fire on Tanya in range. Tanya's HP comfortably absorbs the transit damage; the question is whether you commit to the building or get drawn into a pistol trade. Demolish the refinery within about 30 turns.",
    "≥1 enemy building destroyed, within 2700 ticks.",
    "Deadline (2701 ticks).",
    30, 2703,
    'static (st2 e1 cover on approach)',
    'NOTE: defenders fire on Tanya in range; trade-vs-commit choice. Intended.')


# ── 5. spec-thief-steal-cash — thief cash drain from enemy proc/silo
P = 'spec-thief-steal-cash'; C = 'action'
T = 'observe, move_units, attack_unit, infiltrate'
add(P, 'easy', C, 'spec-thief-chokepoint-40x20', '40x20', 'fit', T,
    '1× thief (unarmed value-extraction specialist; drains cash from an enemy refinery or silo on adjacency, then is consumed) @(16,10)',
    '1× Ore Refinery @(20,10) at the corridor exit; 1× enemy Construction Yard @(35,4) (engine keep-alive); enemy starts with $3000',
    "Commander, this is a value-extraction raid. You are given one thief — an unarmed specialist whose only useful action is to walk into an enemy Ore Refinery or Ore Silo and drain a chunk of their cash into your wallet. The thief is consumed by the steal. A water-walled chokepoint splits the arena; one corridor at y=10 connects west to east, with the enemy Ore Refinery at the corridor exit (x=20). The depot is undefended. Drain enough cash to reach $400 within about 15 turns; the thief carries no weapon and an attack order is a no-op.",
    "Own cash ≥400, within 1350 ticks.",
    "Deadline (1351 ticks), or no thief left.",
    15, 1353,
    'none (undefended depot)', '')
add(P, 'medium', C, 'spec-thief-chokepoint-40x20', '40x20', 'fit', T,
    '1× thief @(16,10)',
    '1× Power Plant @(18,10) (closer; NOT a cash store — infiltrating it wastes the thief) + 1× Ore Silo @(20,10) (the actual depot) + 1× enemy Construction Yard @(35,4); enemy starts with $3000',
    "Commander, this is a value-extraction raid with a distractor target. You are given one thief — an unarmed specialist whose only useful action is to walk into an enemy Ore Refinery or Ore Silo and drain a chunk of their cash into your wallet. The thief is consumed by the steal. Two enemy buildings sit along the corridor: a Power Plant closer to the thief and an Ore Silo one step further. Only the Ore Silo holds cash; infiltrating any other building consumes the thief for nothing. Drain enough cash to reach $400 within about 20 turns.",
    "Own cash ≥400, within 1800 ticks.",
    "Deadline (1801 ticks), or no thief left.",
    20, 1803,
    'none (undefended)',
    'MINOR LEAK: briefing names the Power Plant as wrong-target ("only the Ore Silo holds cash"); could be softened. Acceptable for now (model still must pick).')
add(P, 'hard', C, 'spec-thief-chokepoint-40x20', '40x20', 'fit', T,
    '2× thief @(16,9)/(16,11) — both must reach the depot',
    '1× Ore Silo @(20,10) at the corridor exit + 1× enemy rifleman @(19,10) stance:2 (Defend — fires on thieves on approach, ~25% HP per pass) + 1× enemy Construction Yard @(35,4); enemy starts with $3000',
    "Commander, this is a value-extraction raid against a defended depot. You are given two thieves — unarmed specialists whose only useful action is to walk into an enemy Ore Refinery or Ore Silo and drain a chunk of their cash into your wallet. Each thief is consumed by its own steal, and each yields at most $500. An enemy Ore Silo sits at the corridor exit, guarded by one enemy rifleman who will fire on the thieves as they approach (about 25% HP lost per transit; both thieves survive a single pass each). Drain enough cash to reach $800 within about 30 turns — BOTH thieves must arrive and drain.",
    "Own cash ≥800, within 2700 ticks.",
    "Deadline (2701 ticks), or no thief left.",
    30, 2703,
    'static (st2 e1 at depot)', '')


# ── 6. rush-hour — historical baseline search-and-destroy sweep
P = 'rush-hour'; C = 'action'
T = 'move_units, attack_unit, attack_move, stop_units'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'fit', T,
    '4 corner armies (NW, NE, SW, SE), 6 units each = 24× medium tank + 8× jeep total; all HOLD-FIRE',
    '22× rifle infantry scattered across the 128x40 arena in 11 small groups on three bands (y=9 north, y=19-20 mid, y=31-32 south), all hidden by fog; stance:1 (ReturnFire)',
    "Commander, this is a reconnaissance-and-destroy sweep across the whole 128x40 arena. You are given four corner armies — 24 medium tanks and 8 jeeps in total (six units per corner). The enemy is 22 rifle infantry scattered across the arena in small groups, all hidden by fog of war on the north, mid, and south rows. Spread the armies, reveal the fog as you move, find and destroy the enemy. Score at least seven kills, reveal at least 18% of the map, lose no more than five of your own, within about 41 turns. Sitting still or marching down a single row fails both bars.",
    "≥7 kills AND explored ≥18% AND ≤5 losses, within 3603 ticks.",
    "Force wipeout, >5 lost, or deadline (3603 ticks).",
    41, 3693,
    'reactive (st1 ReturnFire; scattered, hidden by fog)',
    'BASELINE: historical reference pack — do NOT edit unless explicitly invoked.')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'fit', T,
    '4 corner armies (NW, NE, SW, SE), 24× medium tank + 8× jeep total; all HOLD-FIRE',
    '22× rifle infantry scattered across 128x40 in 11 small groups on three bands; stance:1 (ReturnFire); hidden by fog',
    "Commander, this is a wider reconnaissance-and-destroy sweep. You are given four corner armies — 24 medium tanks and 8 jeeps in total. The enemy is 22 rifle infantry scattered across the 128x40 arena in small groups, all hidden by fog of war on the north, mid, and south rows. Spread the armies, reveal the fog, and find and destroy the enemy. Score at least nine kills, reveal at least 28% of the map, lose no more than four of your own, within about 41 turns. A single-row pass tops out at eight kills — clearing both the north and south bands is required.",
    "≥9 kills AND explored ≥28% AND ≤4 losses, within 3603 ticks.",
    "Force wipeout, >4 lost, or deadline.",
    41, 3693,
    'reactive (st1 ReturnFire; scattered, hidden by fog)',
    'BASELINE: do NOT edit unless explicitly invoked.')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'fit', T,
    'Team round-robins NW/NE/SW/SE by seed (4 spawn groups); 24× medium tank + 8× jeep total; all HOLD-FIRE; relative-coordinate briefing',
    '22× rifle infantry scattered across 128x40 in 11 small groups on three bands; stance:1 (ReturnFire); hidden by fog',
    "Commander, this is a reconnaissance-and-destroy sweep across the whole 128x40 arena. Your team round-robins between four spawn corners by seed (compass-only briefing): you may stage at any of NW, NE, SW, or SE — read your own starting corner from the map. You are given 24 medium tanks and 8 jeeps in total (six units per corner). The enemy is 22 rifle infantry scattered across the arena in small groups, all hidden by fog of war on the north, mid, and south rows; localise them on the minimap yourself. Score at least ten kills, reveal at least 32% of the map, lose no more than four of your own, within about 41 turns. Only a real spread across all three rows clears the bar.",
    "≥10 kills AND explored ≥32% AND ≤4 losses, within 3603 ticks.",
    "Force wipeout, >4 lost, or deadline.",
    41, 3693,
    'reactive (st1 ReturnFire; scattered, hidden by fog)',
    'BASELINE: do NOT edit unless explicitly invoked.')


# ── 7. custom-map-no-enemy — pure navigation, no adversary
P = 'custom-map-no-enemy'; C = 'perception'
T = 'move_units, stop'
add(P, 'easy', C, 'singles-maginot (custom confined map)', 'custom', 'fit', T,
    '1× jeep @(8,12) + 2× rifle infantry @(9,14)',
    'NONE — pack has no enemy actors (whitelisted in tests/test_hard_tier.py::_NO_ENEMY)',
    "Commander, this is a pure navigation task on a confined custom map. There is no enemy on this map; the only failure mode is missing the deadline. You are given one jeep and two riflemen on the far west of the playable area. Move any one of your units into the goal zone near (55,16) at the far east of the playable area, within about 10 turns. The fast jeep alone is enough; idling, stopping short, or wandering loses on the clock.",
    "≥1 unit inside radius-5 of (55,16), within 900 ticks.",
    "Deadline (901 ticks), or all units lost (no enemy ⇒ cannot happen).",
    14, 1263,
    'no enemy', 'INTENTIONAL: pure-navigation cell — no adversary, no combat.')
add(P, 'medium', C, 'singles-maginot (custom confined map)', 'custom', 'fit', T,
    '1× jeep @(8,12) + 2× rifle infantry @(9,14)',
    'NONE',
    "Commander, this is a pure navigation task on a confined custom map — a full-force rendezvous. There is no enemy; the only failure mode is missing the deadline. You are given one jeep and two riflemen on the far west of the playable area. Every unit — not just the fast scout — must reach the goal zone at (72,16) at the far east of the playable area, within about 17 turns. Sending only the jeep ahead does not satisfy the all-in-zone requirement; the whole squad must arrive.",
    "ALL units inside radius-5 of (72,16), within 1500 ticks.",
    "Deadline (1501 ticks).",
    20, 1803,
    'no enemy', 'INTENTIONAL: pure-navigation cell.')
add(P, 'hard', C, 'singles-maginot (custom confined map)', 'custom', 'fit', T,
    'sp0 @(8,12)+(9,14) NORTH or sp1 @(8,28)+(9,26) SOUTH by seed: 1× jeep + 2× rifle infantry per spawn group',
    'NONE',
    "Commander, this is a pure navigation task on a confined custom map with seed-rotated staging and a relative-coordinate goal. There is no enemy; the only failure mode is missing the deadline. Your spawn latitude flips by seed (north or south), and the goal zone is described by direction only — at the far east-central edge of the playable area; localise it on your minimap yourself. Every unit — not just the fast scout — must reach the goal zone within about 22 turns.",
    "ALL units inside radius-7 of (72,24), within 1600 ticks.",
    "Deadline (1601 ticks).",
    22, 1983,
    'no enemy', 'INTENTIONAL: pure-navigation cell — relative-coords + seed-rotated spawn.')


# ── Emit CSV ────────────────────────────────────────────────────────
def main():
    fieldnames = [
        'pack', 'level', 'capability', 'map_name', 'map_size',
        'map_fit', 'tools', 'agent_force', 'enemy_force',
        'enemy_posture', 'posture_issue', 'briefing_RA',
        'win_condition', 'lose_condition', 'max_turns', 'tick_budget',
    ]
    with OUT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for row in R:
            w.writerow({k: row.get(k, '') for k in fieldnames})
    print(f"wrote {len(R)} rows to {OUT}")


if __name__ == '__main__':
    main()
