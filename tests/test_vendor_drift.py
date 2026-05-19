"""The bench prompt/briefing/minimap MUST stay byte-identical to the
training rollouts (that's the whole point of vendoring). When the
training checkout is present, byte-compare; skip otherwise so the
bench stays self-contained for CI without it."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

VENDOR = Path(__file__).parent.parent / "openra_bench" / "_vendor"
TRAIN = Path(
    os.environ.get("OPENRA_TRAINING_REPO", "/Users/berta/Projects/OpenRA-RL-Training")
)

PAIRS = {
    "system_v2.txt": TRAIN / "openra_rl_training/prompts/system_v2.txt",
    "briefing_v2.py": TRAIN / "openra_rl_training/prompts/briefing_v2.py",
    "minimap_v2.py": TRAIN / "scripts/_minimap_v2.py",
}


def _sha(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,upstream", list(PAIRS.items()))
def test_vendored_matches_upstream(name, upstream):
    if not upstream.exists():
        pytest.skip(f"training checkout absent ({upstream})")
    vend = VENDOR / name
    assert vend.exists(), f"vendored {name} missing"
    assert _sha(vend) == _sha(upstream), (
        f"{name} drifted from upstream — re-vendor (do not hand-edit)"
    )


def test_vendored_artifacts_present_and_usable():
    import importlib.util as iu

    assert (VENDOR / "system_v2.txt").read_text().rstrip().endswith(
        "OBJECTIVE (this scenario): {objective}"
    )
    for mod in ("briefing_v2", "minimap_v2"):
        spec = iu.spec_from_file_location(mod, VENDOR / f"{mod}.py")
        m = iu.module_from_spec(spec)
        spec.loader.exec_module(m)
    assert hasattr(m, "render")  # minimap_v2 last
