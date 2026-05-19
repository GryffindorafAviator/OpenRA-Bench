"""YAML-referenceable map generator (mapgen).

A pack can declare `base_map: {generator: arena, ...}` and the bench
materializes a real, deterministic .oramap the loader/engine resolve
like any shipped map.
"""
from __future__ import annotations

import zipfile

import pytest

from openra_bench.mapgen import materialize, resolve_base_map, spec_id
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level, resolve_map_path


def test_arena_materializes_and_resolves():
    s = {"generator": "arena", "width": 160, "height": 72, "cordon": 4}
    mid = materialize(s)
    assert mid == spec_id(s) == materialize(s)        # deterministic
    p = resolve_map_path(mid)
    assert p is not None and p.is_file()
    y = zipfile.ZipFile(p).read("map.yaml").decode()
    assert "MapSize: 160,72" in y and "Bounds: 4,4,152,64" in y
    assert {"map.yaml", "map.bin", "map.png"} <= set(
        zipfile.ZipFile(p).namelist()
    )


def test_content_addressed_and_idempotent():
    a = {"generator": "arena", "width": 128, "height": 64}
    b = {"generator": "arena", "height": 64, "width": 128}  # key order
    assert spec_id(a) == spec_id(b)                    # order-independent
    p = resolve_map_path(materialize(a))
    before = p.read_bytes()
    materialize(b)                                     # rewrite same spec
    assert p.read_bytes() == before                    # byte-identical


def test_named_spec_gives_readable_id():
    s = {"generator": "arena", "name": "tiny-test-arena",
         "width": 64, "height": 48, "cordon": 2}
    assert materialize(s) == "tiny-test-arena"
    with pytest.raises(ValueError):
        spec_id({"generator": "arena", "name": "bad name!"})


def test_unknown_generator_and_bounds_rejected():
    with pytest.raises(ValueError):
        resolve_base_map({"generator": "does-not-exist"})
    with pytest.raises(ValueError):
        materialize({"generator": "arena", "width": 9999, "height": 40})
    with pytest.raises(ValueError):
        materialize({"generator": "arena", "width": 64, "height": 64,
                     "cordon": 40})


def test_string_base_map_passthrough():
    assert resolve_base_map("rush-hour-arena") == "rush-hour-arena"


def test_pack_with_generator_spec_compiles():
    # #2 hard dogfoods the YAML path: base_map is a generator spec.
    p = load_pack(PACKS_DIR / "action-sequenced-execution.yaml")
    hard = compile_level(p, "hard")
    assert hard.scenario.base_map == "scout-arena"
    assert hard.map_supported
