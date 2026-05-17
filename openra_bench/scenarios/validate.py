"""`python -m openra_bench.scenarios.validate [pack.yaml | dir]`

Validates pack schema, all three levels' engine compilation, and
win/fail-condition grammar. Exits non-zero on the first failure so it
can gate CI / PRs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .loader import PACKS_DIR, compile_level, is_map_supported, load_pack


def _validate_one(path: Path) -> list[str]:
    errs: list[str] = []
    try:
        pack = load_pack(path)
    except Exception as e:  # noqa: BLE001
        return [str(e)]
    for level in ("easy", "medium", "hard"):
        try:
            compile_level(pack, level)  # constructs ScenarioDefinition + WinCondition
        except Exception as e:  # noqa: BLE001
            errs.append(f"[{pack.meta.id}:{level}] {e}")
    if not is_map_supported(pack.base_map):
        errs.append(
            f"[{pack.meta.id}] base_map {pack.base_map!r} not Rust-loadable yet "
            f"(schema-valid; runner will skip until Phase 3)"
        )
    return errs


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else PACKS_DIR
    files = (
        [target]
        if target.is_file()
        else [p for p in sorted(target.glob("*.yaml")) if not p.name.startswith(("_", "TEMPLATE"))]
    )
    if not files:
        print(f"no pack files found at {target}")
        return 1
    failed = False
    for f in files:
        errs = _validate_one(f)
        warns = [e for e in errs if "not Rust-loadable" in e]
        hard = [e for e in errs if e not in warns]
        status = "FAIL" if hard else ("WARN" if warns else "OK")
        print(f"{status:4}  {f.name}")
        for e in hard:
            print(f"      ✗ {e}")
        for w in warns:
            print(f"      ! {w}")
        failed |= bool(hard)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
