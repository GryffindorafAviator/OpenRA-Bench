"""Builds audits/production_tech_audit.csv — construction / production /
tech-tree audit. One row per pack (not per level).

Only packs where the player BUILDS or DEPLOYS something at run time are
included (i.e. pack's `tools:` exposes `build`/`place_building`/`deploy`/
`start_production`, OR the briefing/win-condition references a build).

Columns per spec at audits/PRODUCTION_TECH_AUDIT.md:
    pack, family, buildables_required, tech_gates_present,
    tech_gates_missing, afford_at_start, afford_by_deadline,
    build_in_budget, faction, issues

Most fields are computed mechanically from each pack's YAML:
  - faction          → base.agent.faction (default allies)
  - tech_gates_present → agent-owned buildings @ EASY level start
                         (scheduled_events spawns count too)
  - afford / build-in-budget → cheapest intended chain vs starting_cash
                         + (income_est × max_turns) and serialised
                         build-seconds vs max_turns
  - tech_gates_missing → cross-reference per the tech tree in
                         audits/PRODUCTION_TECH_AUDIT.md

`buildables_required` requires interpretation of the briefing and the
win-condition predicates. The BUILDABLES table below records the audit
author's reading of each pack; it is the ONLY hand-curated input.

A pack row is emitted with `issues != ""` when any of:
  - tech_gates_missing is non-empty (prereq DEFECT)
  - afford_at_start = False AND afford_by_deadline = False
  - build_in_budget = False
  - faction mismatch (agent: ussr but buildables include Allies-only units)
"""
from __future__ import annotations

import csv
import os
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Static reference data — from audits/PRODUCTION_TECH_AUDIT.md
# ---------------------------------------------------------------------------

BUILDINGS = {
    'fact','tent','barr','powr','apwr','proc','weap','fix','silo','gun',
    'pbox','hbox','tsla','sam','ftur','dome','atek','stek','spen','syrd',
    'afld','hpad','kenn',
}

# Prereqs for each slug. Only the slugs we actually audit are listed; if a
# briefing-derived buildable isn't here it falls through as "no prereq".
# Allies tech tree (the bench default); Soviet differences flagged via
# FACTION_ONLY below.
PREREQS = {
    # infantry
    'e1':  ['tent'],   # rifle
    'e3':  ['tent'],   # rocket
    'e6':  ['tent'],   # engineer
    'e7':  ['tent','fix'],   # Tanya commando
    'thf': ['tent','fix'],   # thief
    # Soviet infantry
    'e2':  ['barr'],
    # vehicles
    'jeep':['weap'],
    '1tnk':['weap'],
    '2tnk':['weap','fix'],   # Allies medium — needs both
    '3tnk':['weap'],         # Soviet heavy
    'mtnk':['weap','fix'],   # Mammoth
    'harv':['weap'],         # harvester
    'mcv': ['weap','fix'],
    # base buildings
    'tent':['fact'],
    'barr':['fact'],
    'powr':['fact'],
    'apwr':['fact'],         # advanced power
    'proc':['fact'],
    'weap':['proc'],
    'fix': ['weap'],
    'silo':['proc'],
    # defenses
    'gun': ['tent','fix'],
    'pbox':['tent'],
    'hbox':['tent','fix'],
    'tsla':['barr'],         # Soviet tesla coil
    # tech / radar / superweapons
    'dome':['proc'],
    'atek':['fix'],
    'stek':['weap'],
    # naval / air (rarely used in bench)
    'spen':['proc'],
    'syrd':['proc'],
    'hpad':['fact'],
    'afld':['fact'],
    # buildings whose prereq we don't audit; treat as no prereq
    'kenn':['fact'],
    'fact':[],
    'sam': ['fact'],
    'ftur':['tent'],
}

FACTION_ONLY = {
    'allies': {'tent','pbox','hbox','gun','2tnk','mtnk','e7','jeep','dome','atek'},
    'soviet': {'barr','tsla','3tnk','e2','kenn'},
}

# Canonical RA costs and build seconds (= turns @ DEFAULT_TICKS_PER_STEP=30).
COST = {
    # Bench-engine canonical costs — verified directly against the
    # `actor!(...)` declarations in openra-sim/src/gamerules.rs and the
    # explicit `assert_eq!(rules.cost("2tnk"), 800)` engine test.
    # Updated 2026-05-24 (P0.2 in PR #30 review): tent 500→400,
    # 2tnk 850→800, pbox 600→400, hbox 800→600, e7 1200→600.
    'e1':100,  'e3':300,  'e6':500,   'e7':600,  'thf':500, 'e2':160,
    'jeep':600,'1tnk':700,'2tnk':800, '3tnk':950,'mtnk':1700,'harv':1400,'mcv':2500,
    'tent':400,'barr':500,'powr':300,'apwr':500,'proc':1400,'weap':2000,'fix':1200,
    'silo':150,'gun':600,'pbox':400,'hbox':600,'tsla':1500,'dome':1000,
    'atek':1500,'stek':2000,'spen':1500,'syrd':1500,'hpad':500,'afld':600,
    'kenn':200,'fact':0,'sam':750,'ftur':600,
}
BUILD_SEC = {
    'e1':5,'e3':8,'e6':10,'e7':30,'thf':15,'e2':6,
    'jeep':14,'1tnk':16,'2tnk':18,'3tnk':20,'mtnk':30,'harv':25,'mcv':40,
    'tent':10,'barr':10,'powr':8,'apwr':12,'proc':28,'weap':30,'fix':22,
    'silo':5,'gun':14,'pbox':12,'hbox':14,'tsla':24,'dome':20,
    'atek':25,'stek':30,'spen':22,'syrd':22,'hpad':10,'afld':14,
    'kenn':6,'fact':0,'sam':16,'ftur':14,
}

BUILD_TOOLS = {'build','place_building','deploy','start_production'}

# ---------------------------------------------------------------------------
# Hand-curated table: what the intended-capability play MUST build for each
# pack. Derived by reading each pack's EASY briefing + win_condition.
# Keep slugs lowercased and comma-listed in the order the chain runs.
# Some economy packs require buying a replacement harv (`harv`) or a
# defender unit (`e1`/`e3`) that won't appear in `win_condition` directly
# but is the only path the briefing supports; that's encoded here.
# ---------------------------------------------------------------------------

BUILDABLES = {
    # build-* family
    'build-defensive-skirt-corners': ['pbox'],
    'build-defensive-tower-cluster': ['pbox'],
    'build-defensive-tower-line':    ['pbox'],
    'build-engineer-rebuild-after-loss': ['powr'],   # rebuild destroyed powr only
    'build-power-online-first':      ['powr','proc'],
    'build-production-throughput-multibuilding': ['weap','2tnk'],
    'build-rally-point-management':  ['e1'],
    'build-sell-and-rebuild-elsewhere': ['proc'],
    'build-sequence-tech-cheapest':  ['powr','proc','weap'],
    'build-sequence-tech-fastest':   ['powr','proc','weap'],
    'build-sequence-tech-most-resilient': ['weap','2tnk'],
    'build-tech-skip-decision':      ['e1'],   # tech-SKIP path: cheap infantry, skip proc/weap/fix
    'building-and-planning':         ['tent','pbox'],
    'def-engineer-repair-under-fire': [],   # proc/fact pre-placed; repair via engineer is the verb
    'def-in-depth':                  ['pbox'],
    'def-in-depth-vs-single':        ['pbox'],
    'def-position-expected-direction': ['pbox'],
    'def-position-revealed-direction': ['pbox'],
    'def-retreat-and-rebuild':       ['fact','proc'],
    'def-surprise-flank-react':      [],   # pbox pre-placed; redeploy + reinforcement only
    'def-tower-line-vs-cluster':     ['pbox'],
    'def-walls-vs-towers':           ['pbox'],
    'def-while-building':            ['pbox'],
    'defense-rush-survive':          ['pbox'],

    # combat-* with builds
    'combat-rocket-soldier-anti-vehicle': ['e3'],
    'combat-vehicle-vs-infantry-counter': ['2tnk'],

    # adv-*
    'adv-rps-counter-pick':          ['2tnk','e3'],
    'perception-count-the-threat':   ['e1'],

    # mcv-* — deploy MCV then build chain
    'mcv-deploy-and-build':          ['fact','powr','tent'],
    'mcv-deploy-defensible-site':    ['fact','powr','tent'],
    'mcv-deploy-near-resource':      ['fact','proc','powr'],
    'mcv-deploy-relocate-under-pressure': ['fact'],
    'mcv-deploy-second-base':        ['fact'],
    'mcv-deploy-third-base':         ['fact'],

    # multi-front bases
    'mfb-base-1-defend-base-2-build': ['fact'],
    'mfb-mirror-base-east-west':     ['fact','proc'],
    'mfb-redundant-tech-buildings':  ['weap','2tnk'],
    'mfb-rotating-production-pressure': ['weap','2tnk'],
    'mfb-tech-base-vs-economy-base': ['proc','weap'],   # proc auto-spawns 2nd harv on place
    'mfb-third-base-against-clock':  ['fact','proc'],
    'mfb-two-base-simultaneous':     ['fact','proc'],

    # econ-* (mostly need harv re-buy or proc expansion)
    'econ-burn-rate-management':     [],   # mature base + tanks pre-placed; discipline only
    'econ-buy-vs-build-decision':    ['2tnk'],   # capex-vs-opex; cheapest is 1+ 2tnks (buy path)
    'econ-cash-reserve-management':  ['weap','harv'],
    'econ-contention-with-enemy':    ['weap','harv'],
    'econ-contested-expansion':      ['proc'],
    'econ-expansion-timing':         ['proc'],
    'econ-mine-and-grow':            ['proc'],
    'econ-multi-patch-allocation':   ['proc'],
    'econ-overflow-to-silos':        ['silo'],
    'econ-quantitative-vs-qualitative-spend': ['2tnk','e3'],
    'econ-recover-from-zero-cash':   ['weap','harv','powr'],
    'econ-replace-dead-harvester':   ['weap','harv'],
    'econ-resource-trade-with-self': ['harv'],
    'econ-second-base-race':         ['proc'],
    'econ-silo-vs-spend':            ['silo'],   # silo is the cheapest of the any_of branch
    'econ-startup-from-scratch':     ['fact','powr','proc'],   # proc auto-spawns harv on place
    'econ-target-cash-amount-by-deadline': ['harv'],
    'econ-tech-vs-expand-decision':  ['proc'],   # or fix path; cheapest is proc
    'economy-force-buildup':         ['e3'],
    'economy-harvest-investment':    ['proc'],
    'economy-harvest-timebox':       ['harv'],
    'economy-investment':            ['proc','powr'],
    'economy-time-box':              ['e1'],
    'expansion-aggro-3-base-greedy': ['fact','fact','fact'],   # deploy 3 MCVs (no cash cost)
    'expansion-balanced-2-base-defended': ['fact','pbox'],
    'expansion-turtle-1-base-fortified': ['pbox','fix','gun'],

    # long-horizon (lh-*) chains
    'lh-100-turn-marathon-survival':   ['pbox'],
    'lh-build-army-coordinate-multifront-attack': ['2tnk'],
    'lh-credit-only-final-phase':      ['2tnk'],
    'lh-defense-tech-second-base':     ['fact','proc','weap','pbox'],
    'lh-econ-army-victory':            ['e1'],   # cheapest: rifle infantry to fill own_units bar
    'lh-multi-checkpoint-5-plus':      ['proc','weap'],   # easy only needs proc+weap; hard adds 2tnk+army
    'lh-opening-to-defense-to-counter':['proc','powr'],
    'lh-opening-to-tech-to-army':      ['proc','weap','2tnk'],
    'lh-progression-stage-locked':     ['powr','proc','weap'],
    'lh-recovery-after-mid-game-loss': ['proc','3tnk'],
    'lh-scout-react-counter':          [],   # pre-placed tanks; build tool is distractor
    'lh-tech-pivot-attack':            ['e3','gun'],
    'lh-tech-rush-vs-army-rush':       ['1tnk'],   # army rush: cheapest tank (no fix needed); briefing offers tech vs army choice
    'longhorizon-opening-to-assault':  ['proc','e1'],   # cheapest: proc + rifle infantry to fill own_units bar

    # maintenance
    'maint-sell-and-recoup-cash':      ['weap','2tnk'],

    # mid-*
    'mid-tech-switch-on-scout':        ['2tnk','e3','gun'],

    # power
    'power-budget-online':             ['proc','weap','fix','2tnk'],

    # proc-* (procedural tests; many do not really need to build, but
    # the tool is exposed as a distractor)
    'proc-only-build-no-combat':       ['weap'],
    'proc-ordered-action-strict':      ['pbox'],
    'proc-strict-toolban-fidelity':    [],   # pre-placed jeeps; build tool is a distractor
    'proc-tool-use-multi-distractor':  [],
    'proc-tool-use-with-distractor':   [],

    # robustness
    'rob-cash-depletion-recovery':     ['proc'],
    'rob-multiple-simultaneous-pressures': ['weap','harv'],
    'rob-objective-change-midway':     [],
    'rob-objective-shift-with-or-clause': [],
    'rob-partial-base-loss-continue':  ['2tnk'],
    'rob-unit-loss-recovery':          ['3tnk'],

    # scout
    'scout-count-defenders':           ['jeep'],
    'scout-jeep-vs-infantry-cost-effective': ['jeep'],

    # strategy / tech
    'strategy-trilemma':               ['weap'],   # cheapest branch of any_of (proc/weap/units)
    'strict-production-bom':           ['barr','e1','e2','e3'],
    'tech-aggro-all-in':               ['proc','weap','tsla'],
    'tech-balanced-econ-then-tech':    ['proc','weap','dome'],
    'tech-production-planning':        ['weap','fix','3tnk'],
    'tech-turtle-defensive-tech':      ['weap','fix','pbox','gun','dome'],
    'tp-survive-and-grow':             ['2tnk'],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / 'openra_bench' / 'scenarios' / 'packs'
OUT   = Path(__file__).parent / 'production_tech_audit.csv'


def load_pack(pid: str):
    fp = PACKS / f'{pid}.yaml'
    with fp.open() as f:
        return yaml.safe_load(f)


def family_of(pid: str) -> str:
    for prefix, fam in [
        ('build-','build'),  ('def-','defense'), ('defense-','defense'),
        ('combat-','combat'),('econ-','economy'),('economy-','economy'),
        ('expansion-','economy'), ('mcv-','build'),('mfb-','multi-base'),
        ('lh-','long-horizon'),('longhorizon-','long-horizon'),
        ('maint-','maintenance'),('mid-','mid-game'),
        ('perception-','perception'),
        ('power-','power'),('proc-','procedural'),('rob-','robustness'),
        ('scout-','scout'),('strategy-','strategy'),('strict-','strict'),
        ('tech-','tech'),('tp-','tempo'),('adv-','adversarial'),
    ]:
        if pid.startswith(prefix):
            return fam
    return 'misc'


def collect_agent_buildings(pack: dict) -> set[str]:
    """Set of agent-owned building slugs at t=0 (easy level), including
    scheduled_events spawns. Per audit spec: prefer EASY level."""
    base = pack.get('base', {}) or {}
    base_actors = base.get('actors', []) or []
    base_sched  = base.get('scheduled_events', []) or []
    levels = pack.get('levels', {}) or {}
    easy_name = 'easy' if 'easy' in levels else next(iter(levels))
    easy = levels[easy_name]
    ov = easy.get('overrides', {}) or {}
    actors = ov.get('actors', base_actors)
    sched  = ov.get('scheduled_events', base_sched)
    out = {a.get('type') for a in actors
           if a.get('owner') == 'agent' and a.get('type') in BUILDINGS}
    for ev in sched or []:
        if ev.get('type') == 'spawn_actors':
            for a in ev.get('actors', []) or []:
                if a.get('owner') == 'agent' and a.get('type') in BUILDINGS:
                    out.add(a.get('type'))
    # If the agent has an MCV they can deploy → fact (treat as present-by-MCV).
    if any(a.get('type') == 'mcv' and a.get('owner') == 'agent'
           for a in actors):
        out.add('fact-via-mcv')
    return out


def has_harv_start(pack: dict) -> bool:
    base = pack.get('base', {}) or {}
    base_actors = base.get('actors', []) or []
    levels = pack.get('levels', {}) or {}
    easy_name = 'easy' if 'easy' in levels else next(iter(levels))
    ov = levels[easy_name].get('overrides', {}) or {}
    actors = ov.get('actors', base_actors)
    return any(a.get('type') == 'harv' and a.get('owner') == 'agent'
               for a in actors)


def starting_cash(pack: dict) -> int:
    levels = pack.get('levels', {}) or {}
    easy_name = 'easy' if 'easy' in levels else next(iter(levels))
    easy = levels[easy_name]
    if 'starting_cash' in (easy.get('overrides') or {}):
        return easy['overrides']['starting_cash']
    return easy.get('starting_cash', pack.get('starting_cash', 0))


def max_turns_easy(pack: dict) -> int:
    levels = pack.get('levels', {}) or {}
    easy_name = 'easy' if 'easy' in levels else next(iter(levels))
    return levels[easy_name].get('max_turns', 0)


def faction_of(pack: dict) -> str:
    return ((pack.get('base', {}) or {}).get('agent', {}) or {}).get('faction', 'allies')


def hard_diffs(pack: dict, easy_bldgs: set[str]) -> set[str]:
    """Return building types present at HARD that AREN'T at easy
    (used to annotate `issues` when hard adds tech gates)."""
    levels = pack.get('levels', {}) or {}
    if 'hard' not in levels: return set()
    base_actors = (pack.get('base', {}) or {}).get('actors', []) or []
    ov = levels['hard'].get('overrides', {}) or {}
    actors = ov.get('actors', base_actors)
    hard = {a.get('type') for a in actors
            if a.get('owner') == 'agent' and a.get('type') in BUILDINGS}
    return hard - {b for b in easy_bldgs if b != 'fact-via-mcv'}


# ---------------------------------------------------------------------------
# Per-row builder
# ---------------------------------------------------------------------------

R = []

def emit(pid: str):
    pack = load_pack(pid)
    fam = family_of(pid)
    faction = faction_of(pack)

    present = collect_agent_buildings(pack)
    # Treat fact-via-mcv as fact-present for tech-gate purposes.
    # For SOVIET packs, `barr` substitutes for the Allies `tent`
    # infantry-building prereq (and vice versa). Same role, different
    # faction sprite — engine treats them as equivalent infantry
    # producers for prereq purposes.
    present_for_prereqs = {b.replace('-via-mcv','') for b in present}
    if 'barr' in present_for_prereqs:
        present_for_prereqs.add('tent')
    if 'tent' in present_for_prereqs:
        present_for_prereqs.add('barr')

    buildables = BUILDABLES.get(pid, [])
    # tech_gates_missing: for every required buildable, its prereqs must
    # be present at t=0 OR appear earlier in the buildables chain.
    # Strict check per the audit spec — transitive "well it's buildable
    # from the chain" is captured by including the prereq IN the chain.
    # barr <-> tent equivalence applies when satisfying as a chain step too.
    chain_equiv = set(buildables)
    if 'barr' in chain_equiv: chain_equiv.add('tent')
    if 'tent' in chain_equiv: chain_equiv.add('barr')

    missing = []
    for b in buildables:
        for pre in PREREQS.get(b, []):
            if pre in present_for_prereqs: continue
            if pre in chain_equiv: continue
            missing.append(f'{b}<-{pre}')
    missing = sorted(set(missing))

    # afford_at_start: cheapest chain cost ≤ starting_cash
    chain_cost = sum(COST.get(b, 0) for b in buildables)
    cash = starting_cash(pack)
    afford_at_start = chain_cost <= cash

    # afford_by_deadline: cash + harv_income * max_turns ≥ chain_cost
    turns = max_turns_easy(pack)
    income_per_turn = 0
    if has_harv_start(pack):
        # Conservative — 95 cr/turn near patch, drop to 50 if there's no
        # nearby proc (we don't measure distance, so use 95 as default).
        income_per_turn = 95
    projected = cash + income_per_turn * turns
    afford_by_deadline = projected >= chain_cost

    # build_in_budget: sum of build seconds ≤ max_turns
    chain_secs = sum(BUILD_SEC.get(b, 0) for b in buildables)
    build_in_budget = chain_secs <= turns

    # Faction mismatch: any buildable that's faction-locked to a different
    # faction than the pack's agent.
    fac_mismatch = []
    for b in buildables:
        for f, allowed in FACTION_ONLY.items():
            if b in allowed and f != faction:
                fac_mismatch.append(f'{b}({f}-only,pack={faction})')

    issues = []
    if missing:
        issues.append(f'missing-prereq:{",".join(missing)}')
    if not afford_at_start and not afford_by_deadline:
        issues.append(f'unaffordable:need={chain_cost},have={cash},proj={projected}')
    elif not afford_at_start:
        issues.append(f'tight-cash:need={chain_cost},have={cash}')
    if not build_in_budget:
        issues.append(f'build-time-over-budget:{chain_secs}s>{turns}t')
    if fac_mismatch:
        issues.append(f'faction-mismatch:{",".join(fac_mismatch)}')
    hd = hard_diffs(pack, present)
    if hd:
        issues.append(f'hard-adds-bldgs:{",".join(sorted(hd))}')

    R.append(dict(
        pack=pid,
        family=fam,
        buildables_required=', '.join(buildables),
        tech_gates_present=', '.join(sorted(present)),
        tech_gates_missing=', '.join(missing),
        afford_at_start=afford_at_start,
        afford_by_deadline=afford_by_deadline,
        build_in_budget=build_in_budget,
        faction=faction,
        issues='; '.join(issues),
    ))


# ---------------------------------------------------------------------------
# Pack list — all packs that use build/place_building/deploy/start_production
# (auto-detected; matches /tmp/buildable_packs.txt at generation time).
# ---------------------------------------------------------------------------

PACK_IDS = [
    'adv-rps-counter-pick',
    'build-defensive-skirt-corners',
    'build-defensive-tower-cluster',
    'build-defensive-tower-line',
    'build-engineer-rebuild-after-loss',
    'build-power-online-first',
    'build-production-throughput-multibuilding',
    'build-rally-point-management',
    'build-sell-and-rebuild-elsewhere',
    'build-sequence-tech-cheapest',
    'build-sequence-tech-fastest',
    'build-sequence-tech-most-resilient',
    'build-tech-skip-decision',
    'building-and-planning',
    'combat-rocket-soldier-anti-vehicle',
    'combat-vehicle-vs-infantry-counter',
    'def-engineer-repair-under-fire',
    'def-in-depth-vs-single',
    'def-in-depth',
    'def-position-expected-direction',
    'def-position-revealed-direction',
    'def-retreat-and-rebuild',
    'def-surprise-flank-react',
    'def-tower-line-vs-cluster',
    'def-walls-vs-towers',
    'def-while-building',
    'defense-rush-survive',
    'econ-burn-rate-management',
    'econ-buy-vs-build-decision',
    'econ-cash-reserve-management',
    'econ-contention-with-enemy',
    'econ-contested-expansion',
    'econ-expansion-timing',
    'econ-mine-and-grow',
    'econ-multi-patch-allocation',
    'econ-overflow-to-silos',
    'econ-quantitative-vs-qualitative-spend',
    'econ-recover-from-zero-cash',
    'econ-replace-dead-harvester',
    'econ-resource-trade-with-self',
    'econ-second-base-race',
    'econ-silo-vs-spend',
    'econ-startup-from-scratch',
    'econ-target-cash-amount-by-deadline',
    'econ-tech-vs-expand-decision',
    'economy-force-buildup',
    'economy-harvest-investment',
    'economy-harvest-timebox',
    'economy-investment',
    'economy-time-box',
    'expansion-aggro-3-base-greedy',
    'expansion-balanced-2-base-defended',
    'expansion-turtle-1-base-fortified',
    'lh-100-turn-marathon-survival',
    'lh-build-army-coordinate-multifront-attack',
    'lh-credit-only-final-phase',
    'lh-defense-tech-second-base',
    'lh-econ-army-victory',
    'lh-multi-checkpoint-5-plus',
    'lh-opening-to-defense-to-counter',
    'lh-opening-to-tech-to-army',
    'lh-progression-stage-locked',
    'lh-recovery-after-mid-game-loss',
    'lh-scout-react-counter',
    'lh-tech-pivot-attack',
    'lh-tech-rush-vs-army-rush',
    'longhorizon-opening-to-assault',
    'maint-sell-and-recoup-cash',
    'mcv-deploy-and-build',
    'mcv-deploy-defensible-site',
    'mcv-deploy-near-resource',
    'mcv-deploy-relocate-under-pressure',
    'mcv-deploy-second-base',
    'mcv-deploy-third-base',
    'mfb-base-1-defend-base-2-build',
    'mfb-mirror-base-east-west',
    'mfb-redundant-tech-buildings',
    'mfb-rotating-production-pressure',
    'mfb-tech-base-vs-economy-base',
    'mfb-third-base-against-clock',
    'mfb-two-base-simultaneous',
    'mid-tech-switch-on-scout',
    'perception-count-the-threat',
    'power-budget-online',
    'proc-only-build-no-combat',
    'proc-ordered-action-strict',
    'proc-strict-toolban-fidelity',
    'proc-tool-use-multi-distractor',
    'proc-tool-use-with-distractor',
    'rob-cash-depletion-recovery',
    'rob-multiple-simultaneous-pressures',
    'rob-objective-change-midway',
    'rob-objective-shift-with-or-clause',
    'rob-partial-base-loss-continue',
    'rob-unit-loss-recovery',
    'scout-count-defenders',
    'scout-jeep-vs-infantry-cost-effective',
    'strategy-trilemma',
    'strict-production-bom',
    'tech-aggro-all-in',
    'tech-balanced-econ-then-tech',
    'tech-production-planning',
    'tech-turtle-defensive-tech',
    'tp-survive-and-grow',
]


def main():
    for pid in PACK_IDS:
        emit(pid)
    fieldnames = [
        'pack','family','buildables_required','tech_gates_present',
        'tech_gates_missing','afford_at_start','afford_by_deadline',
        'build_in_budget','faction','issues',
    ]
    with OUT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in R:
            w.writerow(r)
    # Summary
    n = len(R)
    miss = sum(1 for r in R if r['tech_gates_missing'])
    bad  = sum(1 for r in R
               if not r['afford_at_start'] and not r['afford_by_deadline'])
    bot  = sum(1 for r in R if not r['build_in_budget'])
    print(f'Wrote {n} rows to {OUT}')
    print(f'  packs with missing prereqs : {miss}')
    print(f'  packs unaffordable (both)  : {bad}')
    print(f'  packs build-time over budget: {bot}')


if __name__ == '__main__':
    main()
