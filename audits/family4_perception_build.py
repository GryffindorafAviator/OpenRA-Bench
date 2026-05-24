"""Builds audits/family4_perception.csv — Family-4 (Scout/Perception/Navigation)
manual audit.

One row per (pack, level). Each briefing is SELF-CONTAINED in F1 officer
style — the model sees one level at a time, so every briefing fully
describes the mission framing, the forces given (with positions where
they matter), and the objective from scratch. No "same as before" or
"the same X" references.

Structure per briefing: mission framing → what is given → target/objective.
Red-Alert-specific terms (jeep, fog of war, stance, refinery, etc.) are
explained inline so non-RA readers can follow.

The map_fit column flags scenarios where the map is too large for the
actual decision under test (turning the scenario into search-and-destroy
instead of testing the advertised perception capability). Default: any
pack on rush-hour-arena 128x40 hosting a tight-clock discovery decision
between a west spawn (x≈10) and a far-east target (x≈100+) is
large-trivial.

Scope (19 packs × 3 levels = 57 rows):
- 14 scout-* packs
- 4 perception-* packs
- 1 navigation-* pack
"""
import csv
from pathlib import Path

OUT = Path(__file__).parent / 'family4_perception.csv'
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


# ── 1. scout-and-report — discover far target + full-team return, zero loss
P = 'scout-and-report'; C = 'perception'
T = 'move_units, attack_unit, stop'
add(P, 'easy', C, 'scout-and-report-arena 128x40', '128x40', 'wide', T,
    '2× jeep scouts @(10,20) on Return-Fire stance (auto-fires only after taking hostile fire)',
    '2× rifle infantry pickets across the central east lane @(60,20)/(62,21) HoldFire; far enemy yard @(110,20) + Barracks @(113,22) (discovery target + persistence anchor)',
    "Commander, this is a reconnaissance-and-extract mission. Two jeep scouts stage at the west base (10,20). The discovery target — a specific far enemy yard — sits at the far east (110,20). Two enemy riflemen block the direct east lane near (60,20). The mission has two binding clauses: surface the far yard, AND bring BOTH jeeps back into the (10,20) staging disc. Zero losses allowed — any jeep that dies fails the extract. The clock is about 50 turns.",
    "Discovery surfaces the far yard AND both jeeps inside the (10,20) r=5 disc AND zero losses, within 4500 ticks.",
    "Any jeep lost, no surviving units, or deadline (4501 ticks).",
    52, 4683,
    'pickets st1 ReturnFire — fire only if shot at, no auto-advance',
    'briefing names target coord (110,20) — direction-relative would isolate fog read better; consider trimming explicit coords in image-PRIMARY form')
add(P, 'medium', C, 'scout-and-report-arena 128x40', '128x40', 'wide', T,
    '2× jeep scouts @(10,20) on Return-Fire',
    '4× rifle picket bottleneck (58..62, 16..22) HoldFire + far yard @(110,20) + Barracks @(113,22)',
    "Commander, this is a reconnaissance-and-extract mission against a defended bottleneck. Two jeep scouts at (10,20). The discovery target — a far enemy yard at (110,20). FOUR enemy riflemen form a defended bottleneck across the central lane around x=60, y=16..22, closing the direct east route entirely. Detour through the open bands above y=14 or below y=26, surface the yard, then return BOTH jeeps to the (10,20) disc. Zero losses. About 50 turns.",
    "Discovery surfaces the far yard AND both jeeps inside the (10,20) r=5 disc AND zero losses, within 4500 ticks.",
    "Any jeep lost, no surviving units, or deadline (4501 ticks).",
    52, 4683,
    'pickets st1 ReturnFire',
    '')
add(P, 'hard', C, 'scout-and-report-arena 128x40', '128x40', 'wide', T,
    '2× jeep scouts staged NORTH (10,8) or SOUTH (10,32) by seed',
    'Thickened central bottleneck (6 pickets) + 2 forward inner-band pickets @(50, 12)/(50, 28); far yard @(110,20) + Barracks @(113,22)',
    "Commander, this is a reconnaissance-and-extract mission with seed-rotated staging. Two jeep scouts stage at the NORTH (10,8) or SOUTH (10,32) corner by seed. Discovery target at (110,20). Six rifles thicken the central bottleneck and two flank pickets at x≈50 contest the inner detour bands too. Only the far edge matching your spawn (y≈4 for NORTH, y≈35 for SOUTH) clears the bottleneck. Surface the yard, return BOTH jeeps to the matched corner disc. Zero losses. About 50 turns.",
    "Discovery surfaces the far yard AND both jeeps inside the matching-corner r=5 disc AND zero losses, within 4500 ticks.",
    "Any jeep lost, no surviving units, or deadline (4501 ticks).",
    52, 4683,
    'pickets st1 ReturnFire; intentional',
    '')

# ── 2. scout-and-survive — discover + return alive (no attack tool)
P = 'scout-and-survive'; C = 'perception'
T = 'move_units, stop'
add(P, 'easy', C, 'scout-and-survive-arena 128x40', '128x40', 'wide', T,
    '1× jeep scout @(10,20) Return-Fire (default)',
    '2× e1 pickets on the central lane near (60,20) HoldFire + far yard @(115,20) + Barracks',
    "Commander, this is a reconnaissance-and-survive mission. A lone jeep scout stages at the west base (10,20). The discovery target — an enemy yard — sits at the far east (115,20). Two enemy riflemen block the direct east lane near (60,20). You have NO attack tool; combat is not an option. Drive around the pickets, surface the yard, and bring the jeep BACK to the (10,20) disc. Losing the jeep fails the mission. About 50 turns.",
    "Discovery surfaces the far yard AND the jeep is inside the (10,20) r=5 disc, within 4500 ticks.",
    "Jeep lost, or deadline (4501 ticks).",
    52, 4683,
    'pickets st1 ReturnFire',
    '')
add(P, 'medium', C, 'scout-and-survive-arena 128x40', '128x40', 'wide', T,
    '1× jeep scout @(10,20) Return-Fire',
    '4× rifle bottleneck (58..62, 18..22) HoldFire + far yard @(115,20) + Barracks',
    "Commander, this is a reconnaissance-and-survive mission against a defended bottleneck. One jeep scout at (10,20), no attack tool. The discovery target — an enemy yard at (115,20). FOUR enemy riflemen form a defended bottleneck across the central lane (x=58..62, y=18..22), closing the direct east route. Detour above y<14 or below y>26, surface the yard, return the jeep to the (10,20) disc. About 50 turns.",
    "Discovery surfaces the far yard AND the jeep is inside the (10,20) r=5 disc, within 4500 ticks.",
    "Jeep lost, or deadline (4501 ticks).",
    52, 4683,
    'pickets st1 ReturnFire',
    '')
add(P, 'hard', C, 'scout-and-survive-arena 128x40', '128x40', 'wide', T,
    '1× jeep staged NORTH (10,8) or SOUTH (10,32) by seed',
    '6 central + 2 inner-band rifles; far yard @(115,20) + Barracks',
    "Commander, this is a reconnaissance-and-survive mission with seed-rotated staging. One jeep scout stages at the NORTH (10,8) or SOUTH (10,32) corner by seed; no attack tool. Discovery target at (115,20). Six rifles thicken the central bottleneck; the inner detour bands are contested too. Only the far edge matching your spawn clears safely. Surface the yard, return to the matched corner disc. About 50 turns.",
    "Discovery surfaces the far yard AND the jeep is inside the matching-corner r=5 disc, within 4500 ticks.",
    "Jeep lost, or deadline (4501 ticks).",
    52, 4683,
    'pickets st1 ReturnFire',
    '')

# ── 3. scout-count-defenders — scout the exact K, build K tanks, attack
P = 'scout-count-defenders'; C = 'perception'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop_units'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Full mid-game base: yard + Refinery + Power + War Factory + Service Depot + 2× jeep @(22,19)/(22,21) ReturnFire; $5100',
    'K=2 half-strength medium tanks @(110,20) Defend (60% HP) + 2× pillbox @(108,19)/(108,21) + persistent enemy yard @(120,20)',
    "Commander, this is an exact-count force-sizing drill. You command a full mid-game allied base — Construction Yard, Refinery, Power Plant, War Factory, Service Depot — with two jeep scouts and $5100 cash. The far-east enemy yard is guarded by TWO half-strength medium tanks plus two pillboxes (stationary anti-infantry turrets). The pillboxes don't count toward enemy unit kills — only the tanks do. Build EXACTLY two medium tanks ($800 each) from the war factory and attack-move east. Two enemies discovered, two killed, yard intact, within about 37 turns. Queueing all six tanks before sending misses the deadline.",
    "≥2 enemies discovered AND ≥2 killed AND own yard standing, within 3300 ticks.",
    "Own yard destroyed, or deadline (3301 ticks).",
    38, 3423,
    'enemy 2tnk st2 Defend (hold post, auto-fire in range, no advance); pillboxes hold post; intentional',
    'briefing names exact K=2; on easy this is the rehearsal tier so naming the count is the explicit easy-tier framing — but flagged as a potential leak in image-PRIMARY mode')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same full base + 2× jeep + $5100',
    'K=3 half-strength 2tnk @(110,20) Defend + 1× pillbox + persistent enemy yard',
    "Commander, this is an exact-count force-sizing drill against an unknown enemy count. Same full allied base — yard, refinery, power, war factory, depot — two jeep scouts and $5100. The far-east enemy yard is guarded by an UNKNOWN number of half-strength medium tanks plus a reinforcing pillbox. Drive a jeep east to COUNT the defenders, then build EXACTLY that many medium tanks and attack-move east. Three enemies discovered, three killed, yard intact, within about 42 turns. Under-build loses the trade; over-build (queueing all six) misses the deadline.",
    "≥3 enemies discovered AND ≥3 killed AND own yard standing, within 3700 ticks.",
    "Own yard destroyed, or deadline (3701 ticks).",
    43, 3873,
    'enemy 2tnk st2 + pillboxes hold post; intentional',
    '')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base flips NORTH (y=14) or SOUTH (y=26) by seed; $5100',
    'K=4 half-strength 2tnk @(110,20) + 2× pillbox; persistent enemy yard',
    "Commander, this is an exact-count force-sizing drill with seed-rotated staging. Same full allied base — yard, refinery, power, war factory, depot — but the latitude flips NORTH (y=14) or SOUTH (y=26) by seed. Two jeeps and $5100. An UNKNOWN tank cloud plus two pillboxes guards the eastern yard. Count with the jeeps, build EXACTLY that many medium tanks (up to six fit in the budget), attack-move east. Four discovered, four killed, yard intact, within about 50 turns. Under-build loses the trade; over-build misses the clock.",
    "≥4 enemies discovered AND ≥4 killed AND own yard standing, within 4500 ticks.",
    "Own yard destroyed, or deadline (4501 ticks).",
    52, 4683,
    'enemy 2tnk st2 + pillboxes hold post; intentional',
    '')

# ── 4. scout-cycle-keep-info-fresh — scheduled_events reinforcement
P = 'scout-cycle-keep-info-fresh'; C = 'perception'
T = 'observe, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'scout-arena (rush-hour-shape)', '128x40', 'wide', T,
    '2× jeep @(15, 18)/(15, 22) HoldFire + 3× medium tank @(14-16, 19-21) HoldFire + partial base (yard, refinery, power, barracks, war factory)',
    '4× rocket infantry HoLDING centre @(60,20) HoldFire; mid-episode 3× medium tanks REINFORCE deep in fog @(78,20) at tick 1500; far inert marker',
    "Commander, this is a perception-cycle drill against a static-then-reinforced enemy. Four enemy rocket soldiers hold the centre at (60,20) — they will not advance. Your west base has two jeep scouts and three medium tanks on HOLD-FIRE — the tanks will not engage on their own; you must order them. Mid-episode, around tick 1500, enemy reinforcements arrive DEEP in the fog past the swarm, at a cell your tank attack never visits. A one-shot scout that pulled back misses the reinforcement; the cycle is what counts. See three at first, six total after tick 1500, kill two, within about 45 turns.",
    "Cycle: ≥3 discovered THEN after tick 1500 ≥6 discovered AND ≥2 killed AND own yard standing, within 4000 ticks.",
    "Own yard destroyed, all combat units lost after tick 300, or deadline (4001 ticks).",
    50, 4503,
    'enemy swarm st0 HoldFire + agent st0 HoldFire — intentional; scheduled spawn tick 1500',
    'briefing names reinforce cell (78,20) and tick 1500 — image-PRIMARY mode should redact; consider direction-relative ("deep past the swarm, around mid-episode")')
add(P, 'medium', C, 'scout-arena', '128x40', 'wide', T,
    '2× jeep + 3× medium tank HoldFire + partial base',
    '6× rocket infantry @(60,20) HoldFire + tick-1500 reinforce 3× 2tnk @(78,20); far inert marker',
    "Commander, this is a perception-cycle drill at heavier weight. Six rocket soldiers hold the centre at (60,20). Mid-episode reinforcements arrive deep in the fog past the swarm, around tick 1500. Two jeeps and three tanks on HOLD-FIRE at your west base. The initial swarm alone is short of the post-reinforce bar — only a re-cycled scout sees the new arrivals. Three discoveries initially, eight total after tick 1500, kill three, within about 45 turns.",
    "Cycle: ≥3 discovered THEN after tick 1500 ≥8 discovered AND ≥3 killed AND own yard standing, within 4000 ticks.",
    "Own yard destroyed, all combat units lost after tick 300, or deadline (4001 ticks).",
    50, 4503,
    'enemy swarm st0 + agent st0; intentional',
    '')
add(P, 'hard', C, 'scout-arena', '128x40', 'wide', T,
    'Base flips NORTH (y=14) or SOUTH (y=26) by seed; same 2 jeeps + 3 tanks + partial base',
    'Same 6× rocket centre swarm + tick-1500 reinforce; far inert marker',
    "Commander, this is a perception-cycle drill with seed-rotated staging. Six rocket soldiers hold the centre. Mid-episode reinforcements arrive deep in the fog past the swarm, around tick 1500. Two jeeps and three tanks on HOLD-FIRE at your west base; the base's latitude flips by seed. Cycle scouts out twice — initial sighting, then a re-push to detect the reinforcement — and engage. Three then eight discoveries with the bar after tick 1500, three kills, within about 45 turns.",
    "Cycle: ≥3 discovered THEN after tick 1500 ≥8 discovered AND ≥3 killed AND own yard standing, within 4000 ticks.",
    "Own yard destroyed, all combat units lost after tick 300, or deadline (4001 ticks).",
    50, 4503,
    'enemy swarm st0 + agent st0; intentional',
    '')

# ── 5. scout-deny-enemy-vision — counter-recon, kill the enemy scouts
P = 'scout-deny-enemy-vision'; C = 'reasoning'
T = 'observe, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '2× medium tank @(20, 19-21) HOLD-FIRE',
    '1× enemy rocket-soldier SCOUT @(55,20) HoldFire (the observation post) + far inert marker @(124,20)',
    "Commander, this is a counter-reconnaissance drill. An enemy rocket-soldier scout sits at observation post (55,20), watching your disposition. Your strike force — two medium tanks — holds fire (it will not engage on its own). Drive the force out and KILL the scout before tick 1100. If the scout survives the report window, it has completed its report and you lose. Stalling loses; sweeping the wrong direction loses.",
    "≥1 enemy killed, within 1100 ticks.",
    "Deadline (1101 ticks).",
    25, 2253,
    'enemy scout st0 + agent strike force st0; intentional — explicit attack order is the test',
    '')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '4× medium tank @(20, 11/13/27/29) HoldFire',
    '2× rocket-soldier scouts at SEPARATE posts @(55,12)/(55,28) HoldFire; far inert marker',
    "Commander, this is a counter-reconnaissance drill against two enemy scouts on separate vectors. Two enemy rocket-soldiers hold observation posts — one at the NORTH lane (55,12), one at the SOUTH lane (55,28). Your four medium tanks HOLD-FIRE at the west. Split the force to cover BOTH vectors and kill both scouts before tick 1300 — the report window. Committing the whole force to one lane leaves the other scout to file its report.",
    "≥2 enemies killed, within 1300 ticks.",
    "Deadline (1301 ticks).",
    25, 2253,
    'enemy scouts st0 + agent strike force st0; intentional',
    '')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '8× medium tank staged NORTH (y=9..15) or SOUTH (y=25..31) by seed, all HoldFire',
    '2× rocket-soldier scouts @(55,12)/(55,28) HoldFire (fixed posts); far inert marker',
    "Commander, this is a counter-reconnaissance drill with seed-rotated staging and a tight report window. Two enemy scouts hold observation posts — one at (55,12), one at (55,28). Your eight medium tanks stage at the NORTH or SOUTH base by seed, all on HOLD-FIRE. Split the force to kill both scouts before tick 1100 — a tight window. A memorised intercept plan fails on the flipped seed; committing to one lane lets the other scout file its report.",
    "≥2 enemies killed, within 1100 ticks.",
    "Deadline (1101 ticks).",
    25, 2253,
    'enemy scouts st0 + agent strike force st0; intentional',
    '')

# ── 6. scout-detect-base-direction — scout 2 corners, strike correct
P = 'scout-detect-base-direction'; C = 'perception'
T = 'observe, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'scout-arena', '128x40', 'wide', T,
    '4× medium tank @(10, 18-22) ReturnFire + 2× jeep PRE-STAGED forward at (104,8)/(104,32) HoldFire + own yard @(6,20)',
    'Enemy yards at BOTH corners @(110,8) + @(110,32) (only ONE is the strike target — either raze wins); light e1 picket per corner HoldFire; central-east inert sentinel @(124,20)',
    "Commander, this is a direction-of-target perception drill — visible rehearsal. Two candidate enemy yards sit at the far east corners — NE (110,8) and SE (110,32). Your two jeep scouts are PRE-STAGED forward, near each corner, so both yards are already in line of sight. Four medium tanks stage at the west (10,20). Either yard you raze wins — drive on whichever you reach first. At most one tank loss, within about 28 turns.",
    "Either NE yard razed OR SE yard razed AND ≤1 tank lost AND ≥1 own unit alive, within 2500 ticks.",
    ">1 tank lost, no units alive, or deadline (2501 ticks).",
    50, 4503,
    'corner pickets st0 HoldFire (perception not combat); intentional',
    '')
add(P, 'medium', C, 'scout-arena', '128x40', 'wide', T,
    '4× medium tank + 2× jeep ALL at west base (10,18-22); own yard @(6,20)',
    'Same two corner yards + picket per corner + sentinel; both corners fogged at start',
    "Commander, this is a direction-of-target perception drill with both corners fogged. Two candidate enemy yards at NE (110,8) and SE (110,32), both fogged at start. Four medium tanks and two jeep scouts at your west base. Push the jeeps to confirm which corner actually holds an enemy, then commit the tanks there. Either yard wins. At most one tank loss, within about 28 turns. Blind pre-commit to the wrong corner wastes the deadline.",
    "Either NE or SE yard razed AND ≤1 tank lost AND ≥1 own unit alive, within 2500 ticks.",
    ">1 tank lost, no units alive, or deadline (2501 ticks).",
    50, 4503,
    'corner pickets st0; intentional',
    '')
add(P, 'hard', C, 'scout-arena', '128x40', 'wide', T,
    'Base staged NORTH (y=8) or SOUTH (y=32) by seed; 4 tanks + 2 jeeps + own yard',
    'Same two corner yards + pickets + sentinel + 5× rocket-infantry CURTAIN at x=85, y=16..24 Defend (auto-fires only on units that enter rocket range)',
    "Commander, this is a direction-of-target perception drill with seed-rotated staging and a mid-lane interceptor curtain. Two candidate yards at NE and SE corners, both fogged. Your west staging latitude flips NORTH (y=8) or SOUTH (y=32) by seed. A vertical row of enemy rocket-soldiers holds the middle around x=85, y=16..24 — the on-axis route bypasses them safely, but a cross-axis push drives the tank column through the rocket envelope. Scout both corners, then commit on the latitude-matching corner. At most one tank loss, within about 30 turns.",
    "Either corner yard razed AND ≤1 tank lost AND ≥1 own unit alive, within 2700 ticks.",
    ">1 tank lost, no units alive, or deadline (2701 ticks).",
    45, 4053,
    'corner pickets st0 + curtain st2 Defend (load-bearing: only bites cross-axis columns); intentional',
    'briefing names the curtain coords (x≈85, y≈16..24) — direction-relative "row of rocket soldiers in the middle" reads cleaner in image-PRIMARY')

# ── 7. scout-detect-enemy-tech — discover 3 distinct enemy buildings
P = 'scout-detect-enemy-tech'; C = 'perception'
T = 'move_units, attack_unit, stop'
add(P, 'easy', C, 'scout-detect-tech-arena 128x40 (mid-x split)', '128x40', 'wide', T,
    '2× jeep @(6,18) ReturnFire',
    '2 east-cluster buildings — Construction Yard @(118,18) + War Factory @(114,20); 1× rocket infantry garrison @(116,16) st1',
    "Commander, this is an enemy tech-read rehearsal. Two enemy buildings sit in a tight cluster at the far east — a Construction Yard and a War Factory. Drive both jeeps east, register the pair, halt. Two discoveries within about 25 turns. Brute-attacking the buildings bleeds the jeeps to the garrison; stalling never reaches the cluster.",
    "≥2 buildings discovered, within 2200 ticks.",
    "No surviving units, or deadline (2201 ticks).",
    26, 2343,
    'garrison st1 ReturnFire (fires only if scouts attack); intentional — passive on a fly-by',
    '')
add(P, 'medium', C, 'scout-detect-tech-arena 128x40', '128x40', 'wide', T,
    '2× jeep @(6,18) ReturnFire',
    '3 spaced enemy buildings — yard @(110,18) + War Factory @(116,10) + Radar Dome @(116,28); 2× rocket-soldier garrison HoldFire',
    "Commander, this is an enemy tech-read drill. Three enemy buildings are spaced across the east strip — yard, War Factory, and a tech building (a radar dome reveals advanced units). The buildings sit on different latitudes, so a one-cell nudge or a fixed one-corner push misses one. Split the jeeps — one sweeps the north band, one the south, both cross the middle — to register all three. At most one loss, within about 40 turns.",
    "≥3 buildings discovered AND ≤1 unit lost AND ≥1 alive, within 3600 ticks.",
    ">1 lost, no units alive, or deadline (3601 ticks).",
    40, 3603,
    'garrison st0 HoldFire on passing scouts; intentional',
    '')
add(P, 'hard', C, 'scout-detect-tech-arena 128x40', '128x40', 'wide', T,
    'Jeeps staged NW (6,6) or SW (6,30) by seed, count:2 each group',
    'Enemy yard + War Factory + TWO tech buildings (Tesla NORTH + SAM SOUTH — both placed); 2× rocket-soldier garrison HoldFire',
    "Commander, this is an enemy tech-read drill with seed-rotated staging and split-tech enemy. Three meaningful enemy buildings to identify — yard, War Factory, plus a tech building (Tesla coil or SAM site, both physically placed). Your spawn flips NW (6,6) or SW (6,30) by seed, so the 'near' tech tile changes per run. Discover three distinct buildings with ZERO losses, within about 40 turns. Brute attack wipes the jeeps; a fixed reading order from one spawn misses the seed-placed tech tile from the other.",
    "≥3 buildings discovered AND zero losses AND ≥1 alive, within 3600 ticks.",
    "Any loss, no units alive, or deadline (3601 ticks).",
    40, 3603,
    'garrison st0 HoldFire; intentional',
    '')

# ── 8. scout-detect-incoming-army — early-warning + intercept positioning
P = 'scout-detect-incoming-army'; C = 'perception'
T = 'observe, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '2× jeep @(15,18)/(15,22) HoldFire + 3× medium tank @(14-16, 19-21) AttackAnything',
    '5× rifle infantry @(100,10) AttackAnything (hunt bot — closes on agent); far inert marker @(120,2)',
    "Commander, this is an early-warning intrusion-detection drill. An enemy army is inbound from the NORTH at y=10, spawning at the east edge. Two jeep scouts and three medium tanks are pre-positioned at your west base. Push the scouts forward to localise the threat axis, then re-position the tanks to intercept on the y=10 lane BEFORE the rocket-soldiers close to short range. Four discovered, three killed, at most two losses; at least one tank must be at the forward intercept (around (45,10)). Within about 28 turns.",
    "≥4 discovered AND ≥3 killed AND ≤2 lost AND ≥1 unit at (45,10) r=10, within 2500 ticks.",
    ">2 lost, no units alive, or deadline (2501 ticks).",
    50, 4503,
    'enemy hunt bot st3 (closes on agent); agent jeep st0 (no auto-pull), agent tank st3',
    'agent tank stance:3 is intentional but note CLAUDE.md "stance:3 auto-hunts" — here that drives the intercept once vision is established, which is the test, but a stall could theoretically auto-clear if no other constraint binds; the position clause and the rocket threat keep the bar load-bearing')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '2× jeep + 3× medium tank (same)',
    'Two hunt bands — N @(100,10) and S @(100,30), 4×e1 each, AttackAnything',
    "Commander, this is an early-warning intrusion-detection drill with two uncertain axes. Two enemy bands are inbound — one from NORTH (y=10), one from SOUTH (y=30), each with rifles plus a rocket-soldier, both spawning at the east edge. Two jeeps and three tanks at your west base. Push the scouts on BOTH candidate axes, localise the threat, and commit the tanks to whichever lane closes first. Four discovered, three killed, at most two losses, at least one tank at the matching forward intercept. Within about 28 turns.",
    "≥4 discovered AND ≥3 killed AND ≤2 lost AND ≥1 at (45,10) r=10 OR ≥1 at (45,30) r=10, within 2500 ticks.",
    ">2 lost, no units alive, or deadline (2501 ticks).",
    50, 4503,
    'enemy hunt bot st3; agent st0/st3 mix; intentional',
    '')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base staged NORTH (y=14) or SOUTH (y=26) by seed; same 2 jeeps + 3 tanks',
    'N + S bands, 5×e1 each, AttackAnything',
    "Commander, this is an early-warning intrusion-detection drill with seed-rotated staging. Two enemy bands inbound — NORTH and SOUTH lanes, rifles plus rockets each. Your west base latitude varies by seed. Scout both axes, intercept the closer lane forward. Four discovered, three killed, at most two losses. Within about 28 turns. A memorised opening fails.",
    "≥4 discovered AND ≥3 killed AND ≤2 lost AND ≥1 at the matching forward intercept r=10, within 2500 ticks.",
    ">2 lost, no units alive, or deadline (2501 ticks).",
    50, 4503,
    'enemy hunt bot st3; agent st0/st3 mix; intentional',
    '')

# ── 9. scout-discover-hidden-base — off-axis perception
P = 'scout-discover-hidden-base'; C = 'perception'
T = 'move_units, attack_unit, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '2× jeep @(10,20) HoldFire (no auto-hunt)',
    '2× east decoy rifles @(95,20)/(98,21) HoldFire + HIDDEN enemy yard NW @(10,6) + Barracks @(12,8) (anti-auto-done)',
    "Commander, this is an off-axis reconnaissance drill. Two jeep scouts at (10,20). Two enemy riflemen sit east near (95,20) along the spawn latitude — they LOOK like the enemy presence to investigate, but they're decoys. The REAL enemy base — a Construction Yard — is tucked in the FAR NW corner, BEHIND your spawn. Push NW to discover the real base. At most one loss, within about 40 turns. Sweeping the east decoys or attacking them loses on the clock.",
    "≥1 building discovered AND ≤1 lost, within 3600 ticks.",
    ">1 lost, no units alive, or deadline (3601 ticks).",
    41, 3693,
    'east decoys st0 HoldFire (time-bait, not lethal); agent scouts st0 (no auto-hunt); intentional',
    'briefing names hidden NW coord (10,6) — direction-relative "FAR NW corner behind your spawn" is enough; flag for image-PRIMARY')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '2× jeep @(10,20) HoldFire',
    '3× east decoy rifles + HIDDEN yard deeper NW @(6,5) + Barracks',
    "Commander, this is an off-axis reconnaissance drill with a tighter clock. Two jeeps at (10,20). Three east decoy rifles make a more convincing 'obvious enemy' cluster near (95,20). The REAL enemy yard sits even deeper in the FAR NW corner. The clock won't admit a sweep-east-then-reverse — push off-axis NW immediately. At most one loss, within about 34 turns.",
    "≥1 building discovered AND ≤1 lost, within 3000 ticks.",
    ">1 lost, no units alive, or deadline (3001 ticks).",
    34, 3063,
    'decoys st0 + scouts st0; intentional',
    '')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '2× jeep staged NORTH-leaning (15,16) or SOUTH-leaning (15,24) by seed',
    'Same east decoys + TWO hidden yards (far NW @(5,5) AND far SW @(5,35)) — either discovery wins',
    "Commander, this is an off-axis reconnaissance drill with seed-rotated staging and two real hidden bases. Two jeeps stage at a NORTH-leaning or SOUTH-leaning west centre by seed. East decoys remain a distraction. TWO real hidden yards exist — far NW and far SW. Discover either one; the spawn rotation just controls which corner is the shorter off-axis reach. At most one loss, within about 30 turns.",
    "≥1 building discovered AND ≤1 lost, within 2700 ticks.",
    ">1 lost, no units alive, or deadline (2701 ticks).",
    31, 2793,
    'decoys st0 + scouts st0; intentional',
    '')

# ── 10. scout-far-frontier — pick the fast chassis
P = 'scout-far-frontier'; C = 'perception'
T = 'move_units, stop'
add(P, 'easy', C, 'scout-arena', '128x40', 'wide', T,
    'Own yard @(8,20) + 1× jeep @(12,20) (fast) + 1× rifle @(10,22) (slow)',
    'Far enemy yard @(125,5) + Barracks @(125,8) (anti-auto-done sentinel)',
    "Commander, this is a long-range reconnaissance chassis-pick drill. Your base sits at the west — yard at (8,20) — with two scout assets ready: one fast jeep at (12,20) and one slow rifleman at (10,22). The frontier objective is the far cell (120,30). Get any agent unit within radius 4 of that cell before about turn 17. The jeep crosses in about 9 turns; the rifleman needs about 24 — picking the slow unit or stalling times out.",
    "≥1 unit at (120,30) r=4 AND ≥1 own unit alive AND own yard standing, within 1530 ticks.",
    "No units alive, own yard destroyed, or deadline (1531 ticks).",
    17, 1533,
    'no enemy combat',
    '')
add(P, 'medium', C, 'scout-arena', '128x40', 'wide', T,
    'Own yard + 1× jeep + 1× rifle (same)',
    'Same far enemy yard + Barracks sentinel',
    "Commander, this is a long-range reconnaissance chassis-pick drill at a tighter clock. Same base, same jeep-and-rifleman pair, same frontier at (120,30) — but the clock tightens to about 13 turns. The jeep makes it in roughly 9; the rifleman cannot. Dispatch the jeep or time out.",
    "≥1 unit at (120,30) r=4 AND ≥1 alive AND own yard standing, within 1170 ticks.",
    "No units alive, own yard destroyed, or deadline (1171 ticks).",
    13, 1173,
    'no enemy combat',
    '')
add(P, 'hard', C, 'scout-arena', '128x40', 'wide', T,
    'Base staged NORTH (8,8) or SOUTH (8,32) by seed; own yard + jeep + rifle per group',
    'Per-corner enemy sentinel @(125,3)/(125,37) + central Barracks',
    "Commander, this is a long-range reconnaissance chassis-pick drill with seed-rotated staging. Your base spawns at the NORTH (8,8) or SOUTH (8,32) latitude by seed; the frontier objective tracks the latitude — NE (120,5) for a NORTH start, SE (120,35) for a SOUTH start. Tight clock — about 12 turns. Dispatch the jeep to the SPAWN-MATCHED frontier; the rifleman can't cover the distance, and a wrong-corner commit blows the budget.",
    "≥1 unit at the matching frontier r=4 AND ≥1 alive AND own yard standing, within 1080 ticks.",
    "No units alive, own yard destroyed, or deadline (1081 ticks).",
    12, 1083,
    'no enemy combat',
    '')

# ── 11. scout-jeep-vs-infantry-cost-effective — build the right scout
P = 'scout-jeep-vs-infantry-cost-effective'; C = 'reasoning'
T = 'build, move_units, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Full scout base: yard + Refinery + Power + Barracks + War Factory @(10..18, 18-22); $900',
    'Far frontier enemy yard @(110,20) (sentinel)',
    "Commander, this is a cost-effective recon procurement drill — loose-clock rehearsal. You command an allied scout base — yard, refinery, power, barracks, war factory — and $900 cash. Mission: put a unit on the eastern outpost at (110,20) within radius 5 before about turn 20. The cash buys EITHER one jeep ($600), OR one light tank ($700), OR nine rifles ($100 each) — pick one coherent commit and drive east. The clock is loose; any coherent commit wins.",
    "≥1 unit at (110,20) r=5 AND own yard standing, within 1800 ticks.",
    "Own yard destroyed, or deadline (1801 ticks).",
    21, 1893,
    'no enemy combat',
    '')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same full scout base + $900',
    'Far frontier enemy yard',
    "Commander, this is a cost-effective recon procurement drill with a tight clock. Same scout base — yard, refinery, power, barracks, war factory — and $900. The clock tightens to about 14 turns. The jeep ($600, wheeled, sight 7c) is the only chassis that can field AND traverse the 91-cell run in time — the light tank costs extra travel ticks plus extra build ticks, and rifles need cross-map foot travel after the build. Put one unit in the (110,20) disc with the yard intact, within about 14 turns.",
    "≥1 unit at (110,20) r=5 AND own yard standing, within 1200 ticks.",
    "Own yard destroyed, or deadline (1201 ticks).",
    15, 1353,
    'no enemy combat',
    '')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base flips NORTH (y=14) or SOUTH (y=26) by seed; full scout base + $900 + 1× inert seed-witness rifle HoldFire',
    'Far frontier enemy yard @(110,20) (equidistant from both spawn latitudes)',
    "Commander, this is a cost-effective recon procurement drill with seed-rotated staging. Same scout base, $900, tight clock — but your base spawns NORTH (y=14) or SOUTH (y=26) by seed. Same three composition options (jeep, light tank, nine rifles); only the jeep fields and traverses fast enough. Put one unit in the (110,20) disc with the yard intact, within about 14 turns.",
    "≥1 unit at (110,20) r=5 AND own yard standing, within 1200 ticks.",
    "Own yard destroyed, or deadline (1201 ticks).",
    15, 1353,
    'seed-witness rifle st0 HoldFire (foot too slow to scout, intentional)',
    '')

# ── 12. scout-map-reveal-percent-target — coverage path-planning
P = 'scout-map-reveal-percent-target'; C = 'perception'
T = 'move_units, observe, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'wide', T,
    '4× jeep stacked at SW edge (5, 27..36) Defend (default)',
    'no enemy',
    "Commander, this is a coverage reconnaissance drill. Four jeep scouts at the SW edge. Reveal at least 30% of the map by about turn 6. Bunching the jeeps onto one destination caps coverage on a single swath and just misses the bar; fanning the four jeeps to the four far edges crosses 30% in time. No enemies.",
    "≥30% map explored AND zero losses, within 453 ticks.",
    "Any loss, no units alive, or deadline (454 ticks).",
    8, 723,
    'no enemy combat',
    '')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'fit', T,
    '4× jeep at SW edge (same)',
    'no enemy',
    "Commander, this is a coverage reconnaissance drill at a tighter bar. Four jeep scouts at the SW edge. Reveal at least 50% of the map by about turn 10. A bunched one-swath push tops out near 50% but only at tick 903 — past the deadline. Fan the jeeps to the SE, NE, SW-far, and NW-far edges; the union crosses about 56% in time.",
    "≥50% map explored AND zero losses, within 825 ticks.",
    "Any loss, no units alive, or deadline (826 ticks).",
    12, 1083,
    'no enemy combat',
    '')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'fit', T,
    '5× jeep staged SW column (y=20..36) or NW column (y=4..20) by seed',
    'no enemy',
    "Commander, this is a coverage reconnaissance drill with seed-rotated staging and a high bar. Five jeep scouts stage at the SW or NW edge by seed. Reveal at least 70% of the map by about turn 15. A bunched push asymptotes near 69% and never crosses 70%. Fan to four corners plus a diagonal, then redirect each jeep to the opposite quadrant once its first edge is reached.",
    "≥70% map explored AND zero losses, within 1350 ticks.",
    "Any loss, no units alive, or deadline (1351 ticks).",
    15, 1353,
    'no enemy combat',
    '')

# ── 13. scout-multiple-fog-areas — parallel multi-region scouting
P = 'scout-multiple-fog-areas'; C = 'perception'
T = 'move_units, attack_unit, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'wide', T,
    '3× jeep west column @(5, 14/20/26) Defend',
    '2 region markers — Refinery @(124,4) (NE) + Refinery @(124,36) (SE)',
    "Commander, this is a parallel multi-region scouting drill. Two foggy regions each hide one marker building — the NE corner near (124,4) and the SE corner near (124,36). Three jeeps stage in the west column. Dispatch one jeep to each corner in parallel; a one-corner commit leaves the other in fog. At most one loss, within about 20 turns.",
    "≥2 buildings discovered AND ≤1 lost, within 1800 ticks.",
    ">1 lost, no units alive, or deadline (1801 ticks).",
    22, 1983,
    'no enemy combat (markers unarmed)',
    '')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'fit', T,
    '3× jeep west column (same)',
    '3 markers — Refinery NE (124,4) + SE (124,36) + far-east mid (124,20)',
    "Commander, this is a parallel multi-region scouting drill at a tighter clock. Three foggy regions each hide a marker building — NE corner, SE corner, and far-east mid-band. Three jeeps; one per region in parallel. A serial tour with one jeep can't reach the third before about turn 12. At most one loss.",
    "≥3 buildings discovered AND ≤1 lost, within 1050 ticks.",
    ">1 lost, no units alive, or deadline (1051 ticks).",
    13, 1173,
    'no enemy combat',
    '')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'fit', T,
    '3× jeep staged NW (5, 6..12) or SW (5, 28..34) by seed',
    '4 markers — NE (124,4) + SE (124,36) + N-mid (60,4) + S-mid (60,36)',
    "Commander, this is a parallel multi-region scouting drill with four regions and seed-rotated staging. Three jeeps must sweep four fog regions; one jeep covers two regions on a single edge corridor — NW to NE on the north edge, SW to SE on the south. Your spawn corner round-robins NW/SW by seed; re-derive which jeep sweeps which pair from your actual start row. Four discoveries, one loss max, within about 15 turns.",
    "≥4 buildings discovered AND ≤1 lost, within 1300 ticks.",
    ">1 lost, no units alive, or deadline (1301 ticks).",
    16, 1443,
    'no enemy combat',
    '')

# ── 14. scout-track-enemy-movement — continuous target tracking via scheduled relocate
P = 'scout-track-enemy-movement'; C = 'perception'
T = 'observe, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '2× jeep @(13,18)/(13,22) HoldFire + 4× medium tank @(12, 17-23) HoldFire + partial base (yard + refinery + power + barracks + war factory)',
    'Marching army: 4× rocket-soldier band at LEG A (38,10) at t=0, relocates to LEG B (70,20) at tick 1400, to LEG C (96,30) at tick 2700; far inert marker @(122,4)',
    "Commander, this is a continuous target-tracking drill. An enemy army of four rocket soldiers is on the march — it starts in the NW, relocates to the centre mid-episode, then to the SE later. Two jeeps and four medium tanks at your west base, all on HOLD-FIRE. Keep a scout moving WITH the army through every leg — see it at the NW, then follow it to the centre, then to the SE — so you always know where it is. Then send the tanks to intercept at its final position. Three discoveries at leg 1, six total after the first relocate, nine total after the second, two kills, within about 45 turns.",
    "Track: ≥3 discovered THEN after leg-2 ≥6 discovered THEN after leg-3 ≥9 discovered AND ≥2 killed AND own yard standing, within 4000 ticks.",
    "Own yard destroyed, all combat units lost after tick 300, or deadline (4001 ticks).",
    50, 4503,
    'enemy band st0 HoldFire on every leg + agent combat st0; intentional — scout cycle is the test',
    'briefing names every leg cell + tick — IMAGE-PRIMARY mode this is a near-total leak of the tracking step (the model just drives a jeep to each named cell); flag for rewrite to direction-relative')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '2× jeep + 4× medium tank HoldFire + partial base',
    '6× rocket army marches LEG A (38,10) → LEG B (70,12) at tick 1100 → LEG C (70,28) at tick 2000 → LEG D (96,30) at tick 2900',
    "Commander, this is a continuous target-tracking drill on a FOUR-leg march. An enemy army of six rocket soldiers marches NW → NE around tick 1100 → SE-centre around tick 2000 → SE corner around tick 2900. Two jeeps and four tanks at your west base. Track every leg, then intercept. Four, eight, twelve, sixteen cumulative discoveries, three kills, within about 45 turns.",
    "Track: ≥4 → ≥8 → ≥12 → ≥16 discovered cumulatively with bars between legs AND ≥3 killed AND own yard standing, within 4000 ticks.",
    "Own yard destroyed, all combat units lost after tick 300, or deadline (4001 ticks).",
    50, 4503,
    'enemy band st0 + agent st0; intentional',
    'all four leg coords + ticks named — IMAGE-PRIMARY leak risk')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base flips NORTH (y=14) or SOUTH (y=26) by seed; 2 jeeps + 4 tanks + partial base',
    'Same four-leg march (army doesn\'t honour spawn_point)',
    "Commander, this is a continuous target-tracking drill with seed-rotated staging. Same four-leg march (NW → NE → SE-centre → SE corner) but your west base's latitude flips by seed. Keep a scout moving WITH the army through every leg, then intercept. Four / eight / twelve / sixteen cumulative discoveries, three kills, within about 45 turns. A memorised opening fails.",
    "Track chain (≥4 → ≥8 → ≥12 → ≥16) AND ≥3 killed AND own yard standing, within 4000 ticks.",
    "Own yard destroyed, combat units wiped after tick 300, or deadline (4001 ticks).",
    50, 4503,
    'enemy band st0 + agent st0; intentional',
    '')

# ── 15. perception-count-the-threat — exact-K count + commit
P = 'perception-count-the-threat'; C = 'perception'
T = 'build, place_building, move_units, stop, set_stance, observe'
add(P, 'easy', C, 'perception-count-arena (custom 96x40)', '96x40', 'fit', T,
    'Partial base: yard + Power + Barracks + 1× jeep @(16,20) ReturnFire; $2000',
    '3× rocket-soldier hidden cluster @(60,10) HoldFire + far persistence marker @(90,36)',
    "Commander, this is an exact-count perception drill. Three hidden enemy units sit clustered in a single fog pocket on the north-east half. Your partial base — yard, power, barracks — with one jeep scout and $2000 cash. Scout with the jeep until you have read the count, then build EXACTLY three rifle infantry. Building a fourth busts the over-commit fail; stopping at two never reaches the build target. Within about 30 turns.",
    "≥3 enemies discovered AND own rifle count exactly 3 AND own yard standing, within 2400 ticks (held).",
    "≥4 own rifles built, own yard destroyed, or deadline (2401 ticks).",
    30, 2703,
    'hidden cluster st0 HoldFire; intentional — discovery is free, no auto-engage',
    'briefing names the cluster cell (60,10) — image-PRIMARY mode this hands over the scout step; consider direction-relative')
add(P, 'medium', C, 'perception-count-arena (custom 96x40)', '96x40', 'fit', T,
    'Same partial base + 2× jeep @(16,20)/(16,22); $2400',
    '5 hidden enemy UNITS split (3 NE @(60,10) + 2 SE @(60,30)) HoldFire + enemy silo @(40,20) (building distractor, NOT a unit) + far marker',
    "Commander, this is an exact-count perception drill with unit-vs-building discrimination. Five hidden enemy UNITS spread across two fog pockets — north-east and south-east halves. An enemy ore silo also sits between the clusters — it's a BUILDING, NOT a unit, and must NOT be counted toward the build target. Two jeeps and $2400. Scout both pockets, read the unit count, then build EXACTLY five rifle infantry. A sixth busts the over-commit fail; stopping at four or counting the silo into the build misses the spec. Within about 45 turns.",
    "≥5 enemies discovered AND own rifle count exactly 5 AND own yard standing, within 3600 ticks (held).",
    "≥6 own rifles built, own yard destroyed, or deadline (3601 ticks).",
    45, 4053,
    'hidden clusters st0 + silo neutral; intentional',
    'briefing names both cluster cells + silo cell — image-PRIMARY leak risk')
add(P, 'hard', C, 'perception-count-arena (custom 96x40)', '96x40', 'fit', T,
    '2× jeep at base + $2400',
    'ENEMY-side spawn_point round-robin: K=5 split EITHER 3 NE + 2 SE (sp=0) OR 2 NE + 3 SE (sp=1); 2 building distractors (silo + gun turret) per group + persistence marker',
    "Commander, this is an exact-count perception drill with seed-rotated enemy cluster arrangement. Five hidden enemy units are spread across both halves of the map under fog. Two enemy buildings (an ore silo and a turret) sit on the map — they are NOT units and must NOT be counted. The cluster arrangement rotates per seed — memorising 'always scout north-east' misses two or three of the five on the south-heavy seed. Split the jeeps across both lanes, read the count, build EXACTLY five rifle infantry. A sixth busts the over-commit fail; stopping at four misses the spec. Within about 45 turns.",
    "≥5 enemies discovered AND own rifle count exactly 5 AND own yard standing, within 3600 ticks (held).",
    "≥6 own rifles built, own yard destroyed, or deadline (3601 ticks).",
    45, 4053,
    'enemy clusters st0 + buildings neutral; intentional',
    '')

# ── 16. perception-count-the-threat-small-k — count K=2..4 with "where could the K-th be"
P = 'perception-count-the-threat-small-k'; C = 'perception'
T = 'move_units, attack_unit, stop_units'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '3× jeep @(6,5) HoldFire + 2× light tank @(8,8) HoldFire (all stance:0, NOT auto-hunting)',
    '2 hidden squads — e1 @(55,15) Defend + e3 @(105,18) Defend',
    "Commander, this is a small-K hidden-squad perception drill. Exactly two enemy squads hide in fog — one in the mid-corridor near-east, one in the far-east — on different latitudes. A single straight sweep finds at most one; split the scout column to cover both. Discover both, at most one loss, within about 25 turns.",
    "≥2 enemies discovered AND ≤1 lost, within 2200 ticks.",
    ">1 lost, no units alive, or deadline (2201 ticks).",
    26, 2343,
    'enemy squads st2 Defend (will engage scouts in range); agent scouts st0 HoldFire (must be ordered)',
    'enemy stance:2 means a scout in rocket range gets shot — flagged: per F4 §27 hidden enemies should ideally be st0, but here stance:2 is the documented attrition tooth; OK as long as the briefing prepares the model. Briefing names both cluster cells — image-PRIMARY leak risk')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '3× jeep + 2× light tank HoldFire (same)',
    '3 hidden squads — e1 NE @(105,8) Defend + e3 SE @(105,32) Defend + dog mid-N @(55,4) HoldFire',
    "Commander, this is a small-K hidden-squad perception drill with three squads across three different directions: NE, SE, and a non-obvious mid-north pocket. A naive east-only sweep finds only two before the clock runs out. Read where the unexplored mass is after each contact, then commit there. Discover three, two losses max, within about 32 turns.",
    "≥3 enemies discovered AND ≤2 lost AND ≥1 alive, within 2800 ticks.",
    "No units alive, or deadline (2801 ticks).",
    32, 2883,
    'two squads st2 (attrition tooth) + dog pocket st0 (passive surprise)',
    '')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    '3× jeep + 2× light tank HoldFire (same)',
    '4 hidden squads — 2 obvious east, 1 far-NW dog @(115,4) HoldFire, 1 south-of-mid e2 @(60,28) Defend',
    "Commander, this is a small-K hidden-squad perception drill at the highest K. Four enemy squads: two obvious on the east-bound axes, one behind the spawn in the far-NW strip, one tucked in a tight south-of-mid pocket. Best-effort sweeps find two or three; only inferring from remaining fog mass gets all four. Two losses max, within about 39 turns.",
    "≥4 enemies discovered AND ≤2 lost AND ≥1 alive, within 3500 ticks.",
    "No units alive, or deadline (3501 ticks).",
    40, 3603,
    'armed squads st2 + dog pocket st0; mixed by design',
    '')

# ── 17. perception-frontier-reading — drive scouts into every fog pocket
P = 'perception-frontier-reading'; C = 'perception'
T = 'move_units, attack_unit, stop_units'
add(P, 'easy', C, 'perception-frontier-reading-arena (custom 128x40 with 2 mid-water islands)', '128x40', 'wide', T,
    '5× jeep @(5,33)/(5,33)/(5,33)/(7,30)/(7,30) HoldFire (no auto-advance)',
    '1 hidden marker — Refinery @(108,18) + Barracks @(110,16) (persistence)',
    "Commander, this is a fog-frontier reading drill — rehearsal. One hidden enemy marker sits far east, off your scouts' spawn rows, inside a single contiguous fog mass. Drive a scout into the eastern dark band to surface it. Generous clock, but stalling or sticking near spawn never enters the pocket. Discover at least one enemy building within about 25 turns.",
    "≥1 building discovered, within 2200 ticks.",
    "No units alive, or deadline (2201 ticks).",
    26, 2343,
    'no enemy combat (markers unarmed)',
    '')
add(P, 'medium', C, 'perception-frontier-reading-arena', '128x40', 'wide', T,
    'Same 5 jeeps HoldFire',
    '2 hidden markers — Power @(118,6) + Barracks @(116,8) in the far NE pocket',
    "Commander, this is a fog-frontier reading drill with the markers in a far NE pocket. Two hidden marker buildings sit there, well above your scouts' spawn rows. Both must be discovered. A blind eastward charge along the spawn corridor never reveals either; only a deliberate north-east push reveals both in time. Within about 32 turns.",
    "≥2 buildings discovered, within 2800 ticks.",
    "No units alive, or deadline (2801 ticks).",
    32, 2883,
    'no enemy combat',
    '')
add(P, 'hard', C, 'perception-frontier-reading-arena', '128x40', 'wide', T,
    '5× jeep staged SW (5,33 / 7,30) or NW (5,6 / 7,9) by seed, HoldFire',
    '3 hidden markers — Power @(122,18) + gun turret @(62,4) + Barracks @(116,38); central decoy 3× e1 + 1× e3 @(55-58, 20-22) HoldFire',
    "Commander, this is a fog-frontier reading drill across three corners with a centre decoy. Three fog pockets — far NE, far SE, and a mid-north strip — each hide one marker building. A decoy squad in the seen centre pulls scouts off the true frontier, and your spawn corner varies by seed. Discover all three within about 39 turns. Only a simultaneous read of all three fog pockets fits the clock.",
    "≥3 buildings discovered, within 3500 ticks.",
    "No units alive, or deadline (3501 ticks).",
    40, 3603,
    'decoys st0 HoldFire (time-bait); intentional',
    '')

# ── 18. perception-target-vs-fog — pick which pocket holds the target
P = 'perception-target-vs-fog'; C = 'perception'
T = 'move_units, attack_unit, stop_units'
add(P, 'easy', C, 'perception-target-fog-easy (custom 112x40)', '112x40', 'fit', T,
    '3× jeep @(6,5) HoldFire + 2× light tank @(8,8) HoldFire',
    'NEAR-east DECOY pocket: 2× rifle infantry @(60,6) HoldFire; FAR-east TARGET: enemy yard @(102,8) + rifle guard',
    "Commander, this is a target-among-fog-pockets perception drill. Two unexplored pockets east of your base: a NEAR pocket at the mid-east edge holding only enemy units (no building), and a FAR pocket near the east edge holding the real target building. Sweeping the near pocket alone does not satisfy the win — commit a sustained eastward push past the near pocket to discover the building before about 23 turns.",
    "≥1 building discovered, within 2000 ticks.",
    "Deadline (2000 ticks elapsed) or no units alive.",
    26, 2343,
    'decoys + target guard all st0 HoldFire; intentional — discovery is the test, not combat',
    'briefing names the two pocket coords + which holds the building — moderate leak; the perception step (pick FAR vs NEAR) is named explicitly')
add(P, 'medium', C, 'perception-target-fog-medium (custom 112x40)', '112x40', 'fit', T,
    '3× jeep + 2× light tank HoldFire (same)',
    'NE DECOY @(60,6) units + SE DECOY @(60,33) units + FAR-EAST TARGET enemy yard @(105,20) + guard',
    "Commander, this is a target-among-fog-pockets drill with three candidate pockets. Three unexplored pockets — NE, SE, and FAR-EAST (mid-latitude). NE and SE hold only enemy units (decoys); FAR-EAST holds the real target building. Neither side pocket holds the target — commit a sustained eastward push along the mid-latitude to discover the building before about 26 turns.",
    "≥1 building discovered, within 2300 ticks.",
    "Deadline (2300 ticks elapsed) or no units alive.",
    30, 2703,
    'decoys + target guard st0; intentional',
    'briefing names all three pocket cells + which one holds the target — leak risk in image-PRIMARY')
add(P, 'hard', C, 'perception-target-fog-hard (custom 112x40)', '112x40', 'fit', T,
    '3× jeep + 2× light tank HoldFire (same, fixed across seeds)',
    'ENEMY-side spawn_point 0..3 — the real target yard is at ONE of 4 corner pockets (NE / SE / FAR-NE / FAR-SE) per seed, other 3 carry decoy units only',
    "Commander, this is a target-among-fog-pockets drill with four candidate pockets and seed-rotated target placement. Four candidate pockets at the east-half corners: NE, SE, FAR-NE, FAR-SE. The target building is in one of them; the other three hold only decoy units. WHICH pocket holds the target varies by seed — scout with the jeeps to cover all four corner candidates so the building is discovered before the clock. About 30-turn budget; losing more than one unit busts the cap.",
    "≥1 building discovered AND ≤1 lost, within 2700 ticks.",
    "Deadline (2700 ticks elapsed), >1 lost, or no units alive.",
    34, 3063,
    'decoys + target guard st0 across all 4 spawn groups; intentional',
    'briefing names all four corner coords AND that one holds the target — image-PRIMARY can still infer from the minimap labels; flag for direction-relative')

# ── 19. navigation-confined-hard-only — egress through silo maze
P = 'navigation-confined-hard-only'; C = 'perception'
T = 'move_units, stop'
add(P, 'easy', C, 'singles-maginot', '~120x40', 'wide', T,
    '1× jeep @(8,12) + 2× rifle @(9,14)/(9,15) (no stance set — Defend default)',
    'no enemy',
    "Commander, this is a basic navigation drill on open terrain. No enemy and no walls. Move any unit to the egress zone near (55,16) at the far east of the playable area before about turn 10. Idling or wandering loses.",
    "≥1 unit at (55,16) r=5, within 900 ticks.",
    "No units alive, or deadline (901 ticks).",
    14, 1263,
    'no enemy combat',
    '')
add(P, 'medium', C, 'singles-maginot', '~120x40', 'wide', T,
    '1× jeep + 2× rifle (same)',
    'no enemy',
    "Commander, this is a squad navigation drill on open terrain. Same setup at a longer range: the egress zone sits at (72,16) and EVERY unit must arrive — not just the fast scout. Plan the longer route and march the whole squad. Within about 17 turns.",
    "ALL own units at (72,16) r=5, within 1500 ticks.",
    "No units alive, or deadline (1501 ticks).",
    20, 1803,
    'no enemy combat',
    '')
add(P, 'hard', C, 'confined-aisle-64x40 (custom)', '64x40', 'fit', T,
    '4× rifle staged NORTH (7, 8-14) or SOUTH (7, 25-31) by seed, ReturnFire',
    'Three vertical silo walls @ x=18 / 30 / 42 each with a single 6-cell gap that alternates TOP/BOTTOM/TOP (the serpentine maze); egress marker fact @(55,30)',
    "Commander, this is a confined-aisle egress drill. Three vertical ore silo walls split the playable area into four lanes, each wall with one 6-cell gap that alternates top/bottom/top — the only path is a serpentine detour. Your spawn row is NORTH (y=8..14) or SOUTH (y=25..31) by seed. March every unit into the SE egress zone near (55,30) with zero losses, within about 30 turns. Bashing silos wastes the budget — commit a clean detour.",
    "ALL own units at (55,30) r=6 AND zero losses, within 2700 ticks.",
    "Any loss, or deadline (2701 ticks).",
    31, 2793,
    'silos st0 (inert walls); no enemy combat',
    '')


# ── Emit ─────────────────────────────────────────────────────────────
FIELDS = [
    'pack', 'level', 'capability', 'map_name', 'map_size', 'map_fit',
    'tools', 'agent_force', 'enemy_force', 'enemy_posture',
    'posture_issue', 'briefing_RA', 'win_condition', 'lose_condition',
    'max_turns', 'tick_budget',
]

with OUT.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
    w.writeheader()
    for row in R:
        w.writerow({k: row.get(k, '') for k in FIELDS})

print(f"wrote {OUT} ({len(R)} rows)")
