#!/usr/bin/env python3
"""Consolidate eval playback dirs into the canonical `result/` layout.

Reads existing per-episode playback dirs at:

    data/runs/<out_name>/playback/<run_id>__<model_safe>/<pack>_<level>_<split>/seed<N>/

and copies/symlinks them into:

    result/<run_id>/<model_safe>/<family>/<pack>/<level>/<split>/seed<N>/

plus per-family / per-model / per-run index JSON files. See
`docs/result_schema.md` for the canonical field-level schema.

Usage:
    python3 tools/consolidate_results.py --input data/runs/ --output result/ --symlink
    python3 tools/consolidate_results.py --input data/runs/ --output result/ --copy
    python3 tools/consolidate_results.py --input data/runs/ --output result/ --verify

The script is idempotent: re-running over an existing `result/` is a no-op
for unchanged episodes (files are skipped when target already exists).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Family classification
# ---------------------------------------------------------------------------

# Rule list evaluated in order; first match wins. Each entry is
# (predicate, family-folder-name). The predicates intentionally mirror
# the prefixes catalogued in the task spec.

def _pack_family(pack: str) -> str:
    p = pack.lower()
    # F3 must precede F6 because `build-defensive-*` is a defense pack.
    if p.startswith(("def-", "defense-")) or p.startswith("build-defensive-"):
        return "family3-defense"
    if p.startswith(("combat-", "action-", "harass-")):
        return "family1-combat-micro"
    if p.startswith(("econ-", "economy-")):
        return "family2-economy"
    if p.startswith(("scout-", "perception-", "navigation-")):
        return "family4-scout-perception"
    if p.startswith(("lh-", "longhorizon-")):
        return "family5-long-horizon"
    if p.startswith(("build-", "building-", "tech-", "power-")):
        return "family6-build-tech-power"
    if p.startswith(("proc-", "strict-", "maint-", "rob-")):
        return "family7-procedure-robustness"
    if p.startswith(("mfb-", "mcv-", "coord-", "coordination-")):
        return "family8-multi-front-coord"
    if p.startswith((
        "tp-", "tempo-", "strategy-", "adv-", "adversarial-",
        "artofwar-", "risk-", "reasoning-", "expansion-", "mid-",
    )):
        return "family9-tempo-strategy"
    if p.startswith(("spec-", "custom-")) or p in {"rush-hour"}:
        return "family10-special-misc"
    return "family10-special-misc"  # safe default


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _iter_episode_dirs(input_root: Path) -> Iterable[Path]:
    """Yield every `seed<N>/` dir that contains a manifest.json."""
    for mf in input_root.rglob("manifest.json"):
        # Skip non-canonical layouts. Canonical: .../seed<N>/manifest.json
        if not mf.parent.name.startswith("seed"):
            continue
        yield mf.parent


def _parse_episode_path(seed_dir: Path) -> dict | None:
    """Return {run_id, model_safe, source_root} or None if path is not canonical.

    Canonical: <anything>/<TIMESTAMP>__<MODEL_SAFE>/<EPISODE_DIR>/seed<N>/
    """
    parts = seed_dir.parts
    if len(parts) < 4:
        return None
    seed_name = parts[-1]  # seedN
    _ = parts[-2]  # episode dir
    run_model = parts[-3]  # TIMESTAMP__MODEL_SAFE
    if "__" not in run_model:
        return None
    run_id, _, model_safe = run_model.partition("__")
    if len(run_id) < 13:  # roughly YYYYMMDD-HHMMSS
        return None
    if not seed_name.startswith("seed"):
        return None
    try:
        seed = int(seed_name[4:])
    except ValueError:
        return None
    return {
        "run_id": run_id,
        "model_safe": model_safe,
        "seed": seed,
        "run_model_dir": seed_dir.parent.parent,  # the .../<TS>__<MODEL> dir
    }


# ---------------------------------------------------------------------------
# File materialisation
# ---------------------------------------------------------------------------

# File names that get COPIED in the default ('symlink') hybrid mode —
# lightweight text artifacts that get committed to git. Everything
# else (minimap_turn*.png, messages.json with inlined base64 images)
# gets symlinked to save disk and stays gitignored.
_TEXT_ARTIFACTS = {"manifest.json", "turns.jsonl", "score.json", "_journal.jsonl"}


def _materialize(src: Path, dst: Path, mode: str) -> str:
    """Copy or symlink `src` to `dst`. Returns the action taken.

    `mode` is one of 'symlink' / 'copy'.

    'symlink' mode is HYBRID: small text artifacts (manifest / turns /
    score) are COPIED so the result/ tree is portable and git-stageable
    independently of data/runs/. Heavy artifacts (minimap PNGs,
    messages.json with inlined base64 image data URLs) are SYMLINKED to
    save disk; they are gitignored anyway.

    'copy' mode hard-copies everything.

    Symlink falls back to copy on OSes that reject symlinks (rare on
    macOS/Linux). Symlinks are RELATIVE so result/ is portable as long
    as its sibling data/runs/ moves with it.
    """
    if dst.exists() or dst.is_symlink():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Decide copy-vs-symlink based on the DESTINATION filename — the
    # journal's source name is `_journal__<MODEL>.jsonl` but we want it
    # committed as `_journal.jsonl`.
    if mode == "symlink" and dst.name not in _TEXT_ARTIFACTS:
        try:
            rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
            os.symlink(rel, dst)
            return "symlink"
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(src, dst)
    return "copy"


def _materialize_episode(
    seed_dir: Path,
    target_dir: Path,
    mode: str,
) -> dict:
    """Materialise every file from `seed_dir` into `target_dir`. Returns
    a dict with counts {symlink, copy, skip}."""
    counts = {"symlink": 0, "copy": 0, "skip": 0}
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in seed_dir.iterdir():
        if child.is_dir():
            continue
        action = _materialize(child, target_dir / child.name, mode)
        counts[action] = counts.get(action, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

def _read_manifest(seed_dir: Path) -> dict | None:
    try:
        return json.loads((seed_dir / "manifest.json").read_text())
    except (OSError, ValueError):
        return None


def _read_score(seed_dir: Path) -> dict:
    try:
        return json.loads((seed_dir / "score.json").read_text())
    except (OSError, ValueError):
        return {}


def _summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n_cells": 0, "n_wins": 0, "n_losses": 0,
            "n_draws": 0, "n_errors": 0, "win_rate": 0.0,
            "composite_mean": 0.0, "composite_std": 0.0,
        }
    outcomes = [r.get("outcome") for r in rows]
    n_wins = outcomes.count("win")
    n_losses = outcomes.count("loss")
    n_draws = outcomes.count("draw")
    n_errors = outcomes.count("error")
    comps = [float(r.get("composite", 0.0) or 0.0) for r in rows]
    mean = sum(comps) / n
    var = sum((c - mean) ** 2 for c in comps) / n
    return {
        "n_cells": n,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_draws": n_draws,
        "n_errors": n_errors,
        "win_rate": round(n_wins / n, 4),
        "composite_mean": round(mean, 4),
        "composite_std": round(var ** 0.5, 4),
    }


def _emit_family_index(
    out_dir: Path,
    family: str,
    model_safe: str,
    model_full: str,
    run_id: str,
    episodes: list[dict],
) -> dict:
    """episodes: list of {pack, level, seed, split, outcome, composite, turns}."""
    packs: dict = {}
    for ep in episodes:
        packs.setdefault(ep["pack"], {}).setdefault(ep["level"], {})[str(ep["seed"])] = {
            "outcome": ep.get("outcome"),
            "composite": ep.get("composite"),
            "turns": ep.get("turns"),
            "split": ep.get("split"),
        }
    summary = _summarise(episodes)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "model": model_full,
        "model_safe": model_safe,
        "run_id": run_id,
        "packs": packs,
        "summary": summary,
    }
    (out_dir / "_family_index.json").write_text(json.dumps(doc, indent=2))
    return summary


def _emit_model_index(
    out_dir: Path,
    model_safe: str,
    model_full: str,
    run_id: str,
    families: dict,  # family -> summary dict
    all_eps: list[dict],
) -> dict:
    summary = _summarise(all_eps)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "model": model_full,
        "model_safe": model_safe,
        "run_id": run_id,
        "families": families,
        "summary": summary,
    }
    (out_dir / "_model_index.json").write_text(json.dumps(doc, indent=2))
    return summary


def _emit_run_metadata(
    out_dir: Path,
    run_id: str,
    source_paths: list[str],
    models: list[str],
    families: list[str],
    all_eps: list[dict],
) -> None:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "consolidated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_paths": sorted(set(source_paths)),
        "models": sorted(set(models)),
        "families": sorted(set(families)),
        "summary": _summarise(all_eps),
    }
    (out_dir / "_run_metadata.json").write_text(json.dumps(doc, indent=2))


# ---------------------------------------------------------------------------
# Journal materialisation
# ---------------------------------------------------------------------------

def _find_journal(run_model_dir: Path) -> Path | None:
    """The journal sits at `<playback_root>/_journal__<model_safe>.jsonl`,
    which is the PARENT of `run_model_dir`.

        data/runs/<out>/playback/_journal__<MODEL>.jsonl
        data/runs/<out>/playback/<TS>__<MODEL>/<episode>/seed<N>/

    `run_model_dir` is the `<TS>__<MODEL>` directory; its parent is the
    playback root. We look for any `_journal__*.jsonl` there matching
    the model suffix of `run_model_dir.name`.
    """
    pb_root = run_model_dir.parent
    model_safe = run_model_dir.name.partition("__")[2]
    cand = pb_root / f"_journal__{model_safe}.jsonl"
    if cand.exists():
        return cand
    # Fallback: any _journal__*.jsonl in pb_root
    for p in pb_root.glob("_journal__*.jsonl"):
        return p
    return None


# ---------------------------------------------------------------------------
# Main consolidation
# ---------------------------------------------------------------------------

def consolidate(
    input_root: Path,
    output_root: Path,
    mode: str = "symlink",
    verify: bool = False,
) -> dict:
    # bucket: (run_id, model_safe, family) -> list[(seed_dir, manifest, score)]
    by_bucket: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    # track source playback roots for run metadata
    sources_by_run: dict[str, set[str]] = defaultdict(set)
    models_by_run: dict[str, set[str]] = defaultdict(set)
    discrepancies: list[str] = []

    n_episodes = 0
    for seed_dir in _iter_episode_dirs(input_root):
        info = _parse_episode_path(seed_dir)
        if info is None:
            discrepancies.append(f"non-canonical path: {seed_dir}")
            continue
        mf = _read_manifest(seed_dir)
        if mf is None:
            discrepancies.append(f"unreadable manifest: {seed_dir/'manifest.json'}")
            continue
        pack = mf.get("pack_id")
        level = mf.get("level")
        if not pack or not level:
            discrepancies.append(
                f"missing pack_id/level in manifest: {seed_dir}"
            )
            continue
        # Derive split from the parent dir name `<pack>_<level>_<split>`.
        ep_name = seed_dir.parent.name
        split = "public"
        if ep_name.endswith("_held_out"):
            split = "held_out"
        elif ep_name.endswith("_public"):
            split = "public"
        # else: leave default

        family = _pack_family(pack)
        score = _read_score(seed_dir)
        record = {
            "seed_dir": seed_dir,
            "manifest": mf,
            "score": score,
            "info": info,
            "pack": pack,
            "level": level,
            "split": split,
            "family": family,
        }
        by_bucket[(info["run_id"], info["model_safe"], family)].append(record)
        sources_by_run[info["run_id"]].add(str(info["run_model_dir"]))
        models_by_run[info["run_id"]].add(info["model_safe"])
        n_episodes += 1

    if verify:
        # Verification: just confirm every consolidated target exists.
        missing = []
        for (run_id, model_safe, family), recs in by_bucket.items():
            for r in recs:
                tgt = (
                    output_root / run_id / model_safe / family /
                    r["pack"] / r["level"] / r["split"] /
                    f"seed{r['info']['seed']}"
                )
                if not tgt.exists():
                    missing.append(str(tgt))
        return {
            "verified": len(by_bucket) > 0 and not missing,
            "missing_count": len(missing),
            "missing_sample": missing[:10],
            "discrepancies": discrepancies,
            "n_episodes": n_episodes,
        }

    n_materialized = {"symlink": 0, "copy": 0, "skip": 0}
    per_run_models: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for (run_id, model_safe, family), recs in sorted(by_bucket.items()):
        model_full = (recs[0]["manifest"].get("model") or model_safe)
        family_dir = output_root / run_id / model_safe / family
        family_eps: list[dict] = []
        for r in recs:
            info = r["info"]
            tgt = (
                family_dir / r["pack"] / r["level"] / r["split"] /
                f"seed{info['seed']}"
            )
            counts = _materialize_episode(r["seed_dir"], tgt, mode)
            for k, v in counts.items():
                n_materialized[k] = n_materialized.get(k, 0) + v
            ep = {
                "pack": r["pack"],
                "level": r["level"],
                "seed": info["seed"],
                "split": r["split"],
                "outcome": (
                    r["score"].get("outcome") or r["manifest"].get("outcome")
                ),
                "composite": r["score"].get("composite", 0.0),
                "turns": r["manifest"].get("turns"),
            }
            family_eps.append(ep)
            per_run_models[run_id][model_safe].append({**ep, "family": family})
        fam_summary = _emit_family_index(
            family_dir, family, model_safe, model_full, run_id, family_eps
        )
        # Remember for the per-model index
        per_run_models.setdefault(run_id, {}).setdefault(model_safe, [])
        # store summary in a sidecar
        per_run_models.setdefault(("_fam_summary", run_id, model_safe), []).append(
            {"family": family, **fam_summary}
        )

    # Per-model + per-run indexes
    for run_id, models in per_run_models.items():
        if not isinstance(run_id, str):
            continue  # skip the sidecar sentinel keys
        for model_safe, eps in models.items():
            model_dir = output_root / run_id / model_safe
            fam_summaries = {
                fs["family"]: {k: v for k, v in fs.items() if k != "family"}
                for fs in per_run_models.get(("_fam_summary", run_id, model_safe), [])
            }
            # Try to look up model_full from any manifest
            model_full = model_safe
            for r in by_bucket.get((run_id, model_safe, next(iter(fam_summaries)) if fam_summaries else ""), []):
                model_full = r["manifest"].get("model") or model_safe
                break
            # Find any (run_id, model_safe, *) bucket if the above lookup missed
            if model_full == model_safe:
                for (rid, ms, _fam), recs in by_bucket.items():
                    if rid == run_id and ms == model_safe and recs:
                        model_full = recs[0]["manifest"].get("model") or model_safe
                        break
            _emit_model_index(
                model_dir, model_safe, model_full, run_id, fam_summaries, eps,
            )
            # Copy journal alongside model index
            for (rid, ms, _fam), recs in by_bucket.items():
                if rid != run_id or ms != model_safe or not recs:
                    continue
                jrnl = _find_journal(recs[0]["info"]["run_model_dir"])
                if jrnl is not None:
                    dst = model_dir / "_journal.jsonl"
                    if not dst.exists() and not dst.is_symlink():
                        _materialize(jrnl, dst, mode)
                break

    for run_id in sorted({rid for rid in per_run_models if isinstance(rid, str)}):
        run_dir = output_root / run_id
        all_eps = [
            ep for ms, lst in per_run_models[run_id].items() for ep in lst
        ]
        _emit_run_metadata(
            run_dir, run_id,
            source_paths=sorted(sources_by_run.get(run_id, set())),
            models=[
                (recs[0]["manifest"].get("model") or ms)
                for (rid, ms, _f), recs in by_bucket.items()
                if rid == run_id and recs
            ],
            families=[fam for (rid, _ms, fam) in by_bucket if rid == run_id],
            all_eps=all_eps,
        )

    return {
        "n_episodes": n_episodes,
        "n_buckets": len(by_bucket),
        "n_runs": len({rid for rid in per_run_models if isinstance(rid, str)}),
        "actions": n_materialized,
        "discrepancies": discrepancies,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/runs/",
                    help="root of upstream playback dirs")
    ap.add_argument("--output", default="result/",
                    help="canonical result/ tree to materialise")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--symlink", action="store_const",
                   dest="mode", const="symlink",
                   help="symlink files into result/ (default)")
    g.add_argument("--copy", action="store_const",
                   dest="mode", const="copy",
                   help="hard-copy files into result/")
    g.add_argument("--verify", action="store_true",
                   help="check existing result/ against data/runs/ without writing")
    ap.set_defaults(mode="symlink")
    a = ap.parse_args(argv[1:])

    input_root = Path(a.input).resolve()
    output_root = Path(a.output).resolve()
    if not input_root.exists():
        print(f"input root not found: {input_root}", file=sys.stderr)
        return 1
    output_root.mkdir(parents=True, exist_ok=True)

    if a.verify:
        out = consolidate(input_root, output_root, verify=True)
        print(json.dumps(out, indent=2))
        return 0 if out.get("verified") else 1
    out = consolidate(input_root, output_root, mode=a.mode)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
