# `result/` data layout & schema

This doc is the canonical reference for every artifact emitted by
`openra_bench/run_eval.py` and consolidated under the `result/` tree.
A reader who only sees this file should be able to reconstruct what's
on disk and what each field means.

## Directory layout

```
result/
  <run_id>/                          # e.g. 20260524-053913 (UTC timestamp)
    <model_safe>/                    # e.g. Qwen_Qwen3.5-9B (slashes -> _)
      <family>/                      # one of family1-combat-micro ... family10-special-misc
        <pack>/                      # pack id (e.g. action-multiunit-coordination)
          <level>/                   # easy | medium | hard
            <split>/                 # public | held_out
              seed<N>/               # e.g. seed1
                manifest.json
                messages.json
                turns.jsonl
                score.json
                minimap_turn001.png  # optional; one per turn
                minimap_turn002.png
                ...
        _family_index.json           # per-family summary
      _model_index.json              # per-model summary (all families)
      _journal.jsonl                 # copy of `_journal__<model>.jsonl` (incremental record)
    _run_metadata.json               # run-level summary (all models)
```

### Path component definitions

| Component | Source | Notes |
| --- | --- | --- |
| `<run_id>` | The first 15-char `YYYYMMDD-HHMMSS` prefix of `<timestamp>__<model>` (set by `evaluate(run_id=...)` in `run_eval.py:262`). | Multiple per-family runs that share a launch wave share the same `<run_id>` in their filename (UTC second). |
| `<model_safe>` | `re.sub(r"[^A-Za-z0-9._-]+", "_", model)` (`run_eval.py:264`). | Example: `Qwen/Qwen3.5-9B` -> `Qwen_Qwen3.5-9B`. |
| `<family>` | Derived from `pack` prefix (see `tools/consolidate_results.py::FAMILY_RULES`). | F1 combat / F2 econ / F3 defense / F4 perception / F5 longhorizon / F6 build-tech / F7 procedure / F8 multifront / F9 tempo-strategy / F10 special. |
| `<pack>` | `manifest["pack_id"]`. | Authoritative — the directory name `<pack>_<level>_<split>` produced by `Playback.__init__` is just `manifest.json` fields joined with `_`, so we re-derive from the manifest. |
| `<level>` | `manifest["level"]`. | `easy` / `medium` / `hard`. |
| `<split>` | Last underscore-separated component of the upstream cell directory name. | `public` (the default) or `held_out` (anti-memorization split). Surfaced in the manifest only as part of `manifest["scenario"]` indirectly; the journal carries it explicitly. |
| `seed<N>` | `manifest["seed"]`. | `int`. Hard tier uses 1..4. |

---

## Artifact: `manifest.json`

**Purpose.** One-shot summary of the episode written by
`Playback.finalize(...)` (`openra_bench/playback.py:115`), called from
`run_level(...)` (`openra_bench/eval_core.py:513`). The dictionary
passed in is built inline; every key listed here is unconditionally
present unless flagged optional.

| Field | Type | Units | Required | Provenance |
| --- | --- | --- | --- | --- |
| `scenario` | str | `"<pack_id>:<level>"` | yes | `eval_core.py:515` (formatted from compile result) |
| `pack_id` | str | pack id (e.g. `combat-flanking-attack`) | yes | `compiled.pack_id` (`eval_core.py:516`) |
| `level` | str | `easy` \| `medium` \| `hard` | yes | `compiled.level` |
| `capability` | str | one of `perception` / `reasoning` / `action` (sometimes legacy values like `economy`, `defense`, etc.) | yes | `compiled.meta.capability` |
| `run_id` | str \| null | timestamp, e.g. `20260524-053913` | yes | `playback.run_id` set by `run_eval.py:354` |
| `model` | str \| null | model id, e.g. `Qwen/Qwen3.5-9B` | yes | `playback.model` set by `run_eval.py:354` |
| `seed` | int | RNG seed for this episode | yes | `eval_core.py:521` |
| `outcome` | str | `win` \| `loss` \| `draw` (`error` only appears in the journal, never in a written manifest because no playback is written on error) | yes | `eval_core.py:522` |
| `turns` | int | decision turns actually taken (`<= max_turns`) | yes | `eval_core.py:523` |
| `max_turns` | int | budget cap, `compiled.max_turns` | yes | `eval_core.py:524` |
| `actions_issued` | int | total `Command` objects emitted across all turns | yes | `eval_core.py:525` |
| `actions_warned` | int | engine-rejected/warned commands | yes | `eval_core.py:526` |
| `agent_stats` | object \| null | `{turns, tool_calls, empty_replies}` from `ModelAgent.stats` (`agent.py:663`); null for scripted policies | yes | `eval_core.py:527` |
| `agent_stats.turns` | int | turns the agent observed | – | `agent.py:825` |
| `agent_stats.tool_calls` | int | total tool-calls converted to engine commands | – | `agent.py:883` |
| `agent_stats.empty_replies` | int | turns the model emitted no tool call (auto-fallback) | – | `agent.py:885` |
| `objective_progress` | float | continuous progress toward win condition, [0, 1] | yes | `EpisodeResult.objective_progress` |
| `reward_vector` | object | `{economy, military, territory, scouting, objective}` each float in [0, 1] | yes | `EpisodeResult.reward_vector` (`eval_core.py:529`) |
| `signals` | object | terminal-snapshot of engine signals | yes | `eval_core.py:530` |
| `signals.economy_value` | int | `cash + resources` at episode end | yes | `eval_core.py:531` |
| `signals.explored_percent` | float | fog explored fraction, rounded to 0.01, percent (0..100) | yes | `eval_core.py:533` |
| `signals.units_killed` | int | enemy actor kills credited to agent | yes | `eval_core.py:536` |
| `signals.units_lost` | int | own actors destroyed during episode | yes | `eval_core.py:537` |

Notes:
- `capability` is the *primary* capability axis declared in the pack
  YAML; the scoring sub-scores (perception/reasoning/action) are
  always computed regardless of this label.
- Older / scripted runs may have `model = null` and `agent_stats = null`.

---

## Artifact: `turns.jsonl`

**Purpose.** One JSON object per decision turn (newline-delimited),
plus a final synthetic "episode end" record. Written by
`Playback.record_turn(...)` (`openra_bench/playback.py:61`) from
`run_level(...)` (`openra_bench/eval_core.py:387`, `:488`).

Each line is a JSON object with these keys:

| Field | Type | Units | Required | Provenance |
| --- | --- | --- | --- | --- |
| `turn` | int | 1-indexed decision turn | yes | `playback.py:82` |
| `tick` | int \| null | engine `game_tick` at end of turn (cumulative across episode) | yes | `signals.game_tick` |
| `interrupt` | str \| null | name of the interrupt that paused the env (when interrupts: are declared); `null` in non-interrupt mode | yes | `playback.py:84` |
| `commands` | list[str] | `repr(Command)` strings; e.g. `"Command::MoveUnits { unit_ids: [\"1001\"], target_x: 44, target_y: 4 }"` | yes | `playback.py:85` |
| `signals` | object | snapshot of engine signals at this turn | yes | `playback.py:86` |
| `signals.cash` | int | dollars (in-game currency units) | – | `playback.py:88` |
| `signals.economy_value` | int | `cash + resources` (banked + unsold ore) | – | `playback.py:89` |
| `signals.explored_percent` | float | percent 0..100, rounded to 0.01 | – | `playback.py:91` |
| `signals.units_killed` | int | cumulative enemy kills since episode start | – | `playback.py:94` |
| `signals.units_lost` | int | cumulative own losses since episode start | – | `playback.py:95` |
| `signals.enemies_seen` | int | distinct enemy actor ids observed so far | – | `playback.py:96` (`len(enemies_seen_ids)`) |
| `units` | list[object] | own-unit summary at end of turn | yes | `render_state["units_summary"]` |
| `units[*].id` | str | engine actor id (stringified int) | – | `RustObsAdapter.render_state` |
| `units[*].cell_x`,`cell_y` | int | cell coordinates | – | – |
| `units[*].type` | str | RA actor type (e.g. `2tnk`, `e1`, `jeep`) | – | – |
| `units[*].hp` | float | normalized HP, [0.0, 1.0] | – | – |
| `units[*].activity` | str | e.g. `idle`, `moving`, `attacking` | – | – |
| `units[*].idle` | bool | true iff `activity == "idle"` | – | – |
| `units[*].can_attack` | bool | unit has a weapon | – | – |
| `units[*].target_x`,`target_y` | int | move/attack destination when moving | optional | only present when `activity != "idle"` |
| `enemies` | list[object] | visible enemy summary at end of turn (same per-actor shape as `units[*]` minus `can_attack`/`idle`) | yes | `render_state["enemy_summary"]` |
| `goal` | object | per-turn goal tracker snapshot | yes | `playback.py:103` |
| `goal.leaves` | list[object] | each leaf of the win-condition tree with its current value | – | – |
| `goal.leaves[*].name` | str | predicate name (e.g. `units_in_region_gte`) | – | – |
| `goal.leaves[*].target` | various | the literal target (dict, int, str) | – | – |
| `goal.leaves[*].current` | various \| null | current value (null when not measurable yet) | – | – |
| `goal.leaves[*].ratio` | float | current/target, clipped to [0, 1] | – | – |
| `goal.leaves[*].satisfied` | bool | leaf truth value | – | – |
| `goal.reward_vector` | object | same keys as manifest `reward_vector` | – | – |
| `goal.objective_progress` | float | aggregate progress [0, 1] | – | – |
| `goal.won` | bool | true iff win predicate satisfied at this turn | – | – |

The synthetic final record (written by `eval_core.py:486-491`) carries
`commands = ["(episode end: <outcome>)"]` and a fresh signals/goal
snapshot. Use `turn == max(turn)` or `commands[0].startswith("(episode end")`
to identify it.

---

## Artifact: `score.json`

**Purpose.** Compact scorecard with composite + sub-scores + speed,
written by `run_eval.py:409-424`. Produced from a `ScoreCard` dataclass
(`openra_bench/scoring.py:44`).

| Field | Type | Units | Required | Provenance |
| --- | --- | --- | --- | --- |
| `composite` | float | weighted scalar [0, 1] including speed bonus | yes | `ScoreCard.composite` |
| `outcome` | str | `win` \| `loss` \| `draw` | yes | `ScoreCard.outcome` |
| `perception` | float | sub-score [0, 1] | yes | `ScoreCard.perception` |
| `reasoning` | float | sub-score [0, 1] | yes | `ScoreCard.reasoning` |
| `action` | float | sub-score [0, 1] | yes | `ScoreCard.action` |
| `weakest_link` | str | `perception` \| `reasoning` \| `action` | yes | `ScoreCard.weakest_link` |
| `objective_progress` | float | terminal `EpisodeResult.objective_progress` [0, 1] | yes | `run_eval.py:419` |
| `reward_vector` | object | terminal `EpisodeResult.reward_vector` | yes | `run_eval.py:420` |
| `notes` | list[str] | human-readable provenance notes (e.g. `"won in 8 turns / tick 723 of 2200 (speed 0.67, +0.034 bonus)"`) | yes | `ScoreCard.notes` |

Note: `score.json` is a *subset* of the in-memory `ScoreCard` — fields
like `weights`, `win_tick`, `win_turns`, `win_budget`, `speed`,
`composite_base` are NOT persisted here but ARE persisted in the
journal record.

---

## Artifact: `messages.json`

**Purpose.** Full model<->env chat transcript. Written by
`Playback.write_messages(...)` (`openra_bench/playback.py:108`) from
`ModelAgent.history` (`openra_bench/agent.py`).

Top-level: a JSON array of message objects. The exact mix of roles
depends on the provider's tool-call protocol, but every transcript
contains:

| Role | Required keys | Optional keys | Notes |
| --- | --- | --- | --- |
| `system` | `role`, `content` | – | One system prompt at index 0; carries codex, objective, primer. `content` is a string. |
| `user` | `role`, `content` | – | One per turn (briefing + optional minimap data-URL). `content` is either a plain string OR a list of typed parts `[{type:"text",text:...}, {type:"image_url",image_url:{url:"data:image/png;base64,..."}}]` depending on `fog_mode` / `--no-vision`. |
| `assistant` | `role`, `content`, `tool_calls` | `reasoning` | One per turn. `content` is a string (often empty). `tool_calls` is a list of `{id, type:"function", function:{name, arguments:{...}}}`. Provider-specific `reasoning` (e.g. Anthropic thinking, OpenAI o1 reasoning) is preserved when present. |
| `tool` | `role`, `tool_call_id`, `content` | – | One per tool call in the preceding assistant turn. `content` is a short string ack (e.g. `"ok"`). |

Schema notes:
- The minimap image is inlined as a base64 data URL (`data:image/png;base64,...`). To get just the PNG bytes, use the `minimap_turnNNN.png` sidecar files (same content, decoded).
- `tool_calls[*].function.arguments` is a parsed JSON OBJECT (not the OpenAI-spec string). The bench parses provider output into a stable shape before logging.

---

## Artifact: `minimap_turnNNN.png`

**Purpose.** PNG snapshot of the labelled tactical minimap shown to
the model on turn `NNN` (1-indexed, zero-padded to 3 digits). Written
by `Playback.record_turn(...)` (`openra_bench/playback.py:76`) when
the agent's briefing payload includes a minimap base64 blob.

- Format: PNG.
- Dimensions: variable (depends on `render_tactical_minimap` resolution; usually ~400-800px wide).
- Provenance: base64-decoded from the same blob that goes into `messages.json` for that turn.
- Naming: `minimap_turn001.png`, `minimap_turn002.png`, ... in lockstep with `turns.jsonl`.
- Absent for: scripted-baseline runs (no model), `--no-vision` runs (text-only), `structured` fog modes (no image channel). Always present in the canonical `vision` fog mode.

---

## Artifact: `_journal.jsonl` (a.k.a. `_journal__<model>.jsonl`)

**Purpose.** Append-only run journal for resume + post-hoc reporting.
Written by `openra_bench/resilience.py::RunJournal`, populated by
`run_eval.py::_persist` (`run_eval.py:486`). Lives at the playback
root (one per `(out_dir, model)`), NOT in each episode dir. The
consolidator copies it to `result/<run_id>/<model_safe>/_journal.jsonl`.

Each line is one journal entry. The `_key` field is the resume cursor
(`pack|level|split|seed|fog_mode`).

| Field | Type | Units | Required | Provenance |
| --- | --- | --- | --- | --- |
| `cell` | str | cell id, e.g. `combat-attack-from-behind-fog:medium` or `pack:level:fog` (perception sweep) or `pack:level:handoff-base` (handoff sweep) | yes | `run_eval.py:426` |
| `capability` | str | `compiled.meta.capability` | yes | `run_eval.py:427` |
| `split` | str | `public` \| `held_out` | yes | `run_eval.py:428` |
| `seed` | int | episode seed | yes | – |
| `repeat` | int | 0-indexed; `--repeats N` runs each cell N times for pass^k | yes | `run_eval.py:430` |
| `outcome` | str | `win` \| `loss` \| `draw` \| `error` | yes | – |
| `composite` | float | [0, 1] | yes | – |
| `perception` | float | [0, 1] | yes | – |
| `reasoning` | float | [0, 1] | yes | – |
| `action` | float | [0, 1] | yes | – |
| `weakest_link` | str | `perception` \| `reasoning` \| `action` \| `n/a` (errors) | yes | – |
| `objective_progress` | float | [0, 1] | yes | – |
| `reward_vector` | object | same shape as manifest's | yes | – |
| `turns` | int | decision turns taken | yes | – |
| `speed` | float | [0, 1]; only > 0 on wins | yes | `ScoreCard.speed` |
| `win_turns` | int | turns to reach win (0 if not won) | yes | `ScoreCard.win_turns` |
| `notes` | list[str] | human-readable notes (may include `"objective not met (loss); weakest link: perception"` etc.) | yes | – |
| `passivity` | float \| null | handoff: fraction of model turns spent observe/stop only; `null` for non-handoff cells | yes | `run_eval.py:443` |
| `handoff` | object \| null | handoff stats (`{prefix_turns, kind, ...}`), `null` for non-handoff | yes | `run_eval.py:444` |
| `_key` | str | resume key `pack|level|split|seed|fog_mode` | yes | `resilience.py::episode_key` |

The legacy LOSS-only entries (pre-`f9c9c46`) may omit `composite`/`speed`/`win_turns` and the P/R/A subscores. The consolidator preserves the line verbatim.

---

## Artifact: `_family_index.json`

**Purpose.** Per-(run, model, family) summary written by
`tools/consolidate_results.py`. NOT produced by any eval code path —
emitted at consolidation time.

```json
{
  "family": "family1-combat-micro",
  "model": "Qwen/Qwen3.5-9B",
  "model_safe": "Qwen_Qwen3.5-9B",
  "run_id": "20260524-053913",
  "packs": {
    "<pack>": {
      "<level>": {
        "<seed>": {
          "outcome": "win|loss|draw|error",
          "composite": 0.87,
          "turns": 15,
          "split": "public|held_out"
        }
      }
    }
  },
  "summary": {
    "n_cells": 174,                # total episodes summed
    "n_wins": 84,
    "n_losses": 87,
    "n_draws": 3,
    "n_errors": 0,
    "win_rate": 0.4828,
    "composite_mean": 0.4192,
    "composite_std": 0.21
  }
}
```

## Artifact: `_model_index.json`

**Purpose.** Per-(run, model) summary aggregating across families.
Same schema as `_family_index.json` minus the `family` key; gains a
`families` map.

```json
{
  "model": "Qwen/Qwen3.5-9B",
  "model_safe": "Qwen_Qwen3.5-9B",
  "run_id": "20260524-053913",
  "families": {
    "family1-combat-micro": {"n_cells": 174, "win_rate": 0.48, ...},
    "family2-economy": {...}
  },
  "summary": {"n_cells": 870, "win_rate": 0.41, ...}
}
```

## Artifact: `_run_metadata.json`

**Purpose.** Top-level summary for a run.

```json
{
  "run_id": "20260524-053913",
  "consolidated_at": "2026-05-24T18:00:00Z",
  "source_paths": ["data/runs/family1-v3-qwen9b/playback/20260524-053913__Qwen_Qwen3.5-9B", ...],
  "models": ["Qwen/Qwen3.5-9B", ...],
  "families": ["family1-combat-micro", ...],
  "summary": {
    "n_cells": 1740,
    "n_wins": 720,
    "n_losses": 980,
    "n_draws": 30,
    "n_errors": 10,
    "win_rate": 0.4138
  }
}
```

---

## Versioning & change policy

- Adding a NEW optional field to any artifact is non-breaking; updating
  this doc + the consolidation `summary` block is sufficient.
- Removing or renaming an existing field requires bumping the schema
  version in this doc and bumping the consolidation script's
  `SCHEMA_VERSION`.
- A `score.json`/`manifest.json` field that varies per provider
  (e.g. `agent_stats` keys) must be documented in this file with the
  variation called out — never silently elided.
