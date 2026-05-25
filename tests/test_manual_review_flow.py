"""Human-review-before-save flow for the manual-play UI.

Verifies the three required behaviours of the `manual_review=True`
gate on `InteractiveSession` and the matching FastAPI endpoints:

1. Finalize stages the playback in a `.draft/` dir — the real final
   path stays absent until the human commits.
2. `commit_playback()` (and the `/api/game/commit/{sid}` endpoint)
   promotes the draft to the canonical Playback layout and exposes
   `save_path`.
3. `discard_playback()` (and the `/api/game/discard/{sid}` endpoint)
   removes the draft and the canonical save path never materialises.

The pre-existing scripted/run_level Playback path is non-draft and
covered separately by `tests/test_playback*.py`; this file does NOT
re-test it.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _smallest_easy_pack():
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import PACKS_DIR, compile_level

    best = None
    for f in sorted(PACKS_DIR.glob("*.yaml")):
        if f.name.startswith(("_", "TEMPLATE")):
            continue
        try:
            pack = load_pack(f)
            if pack.meta.status != "active" or "easy" not in pack.levels:
                continue
            c = compile_level(pack, "easy")
        except Exception:  # noqa: BLE001
            continue
        if c.map_supported and (
            best is None or c.max_turns < best.max_turns
        ):
            best = c
    return best


def _drive_to_done(sess, max_extra=5):
    from openra_bench.human_labeling import HumanAction

    guard = sess.max_turns + max_extra
    steps = 0
    while not sess.done and steps < guard:
        sess.submit_turn([HumanAction(mode="observe")])
        steps += 1
    assert sess.done


def _final_seed_dir(playback_root: Path) -> Path | None:
    """The single canonical `<run>__<player>/<cell>/seedN` dir that a
    committed playback should live under. Returns None if no
    non-draft seed dir exists."""
    for run_dir in playback_root.iterdir():
        if run_dir.name == ".draft":
            continue
        for cell_dir in run_dir.iterdir():
            for seed_dir in cell_dir.iterdir():
                if seed_dir.name.startswith("seed"):
                    return seed_dir
    return None


# ── (1) finalize-as-draft does NOT touch the real save_path ──────────


def test_manual_review_finalizes_to_draft_not_final(tmp_path):
    from openra_bench.human_labeling import InteractiveSession

    compiled = _smallest_easy_pack()
    assert compiled is not None

    sess = InteractiveSession(
        compiled, seed=1, record=True, playback_root=tmp_path,
        player="Human", manual_review=True,
    )
    try:
        _drive_to_done(sess)
        # status() must report pending review, NOT a save_path.
        st = sess.status()
        assert st["manual_review"] is True
        assert st["pending_review"] is True
        assert st["save_path"] is None
        # The staged path lives under .draft/.
        draft = st["draft_path"]
        assert draft, "draft_path must be set after finalize"
        draft_p = Path(draft)
        assert draft_p.is_dir()
        assert ".draft" in draft_p.parts
        # The canonical final directory MUST NOT exist yet.
        assert _final_seed_dir(tmp_path) is None
        # The draft IS a full Playback dir — manifest/turns are written
        # so the reviewer sees exactly what would be saved.
        assert (draft_p / "manifest.json").is_file()
        assert (draft_p / "turns.jsonl").is_file()
        assert (draft_p / "messages.json").is_file()
        # preview_turns() reads the staged turns.jsonl.
        rows = sess.preview_turns()
        assert len(rows) == sess.turn
        for r in rows:
            assert "turn" in r and "command_summary" in r
            assert "signals" in r
    finally:
        sess.close()


# ── (2) commit promotes draft → canonical layout, save_path exposed ──


def test_commit_promotes_draft_to_canonical_layout(tmp_path):
    from openra_bench.human_labeling import InteractiveSession

    compiled = _smallest_easy_pack()
    assert compiled is not None

    sess = InteractiveSession(
        compiled, seed=1, record=True, playback_root=tmp_path,
        player="Human", manual_review=True,
    )
    try:
        _drive_to_done(sess)
        draft = Path(sess.status()["draft_path"])
        assert draft.exists()

        save_path = sess.commit_playback()
        assert save_path is not None
        saved = Path(save_path)
        assert saved.is_dir()
        # No more .draft/ remnant for this session.
        assert not draft.exists()
        # Canonical layout is unchanged from the pre-review path —
        # same files a model run produces.
        assert (saved / "manifest.json").is_file()
        assert (saved / "turns.jsonl").is_file()
        assert (saved / "messages.json").is_file()
        assert list(saved.glob("minimap_turn*.png"))
        # status() now reports the published save_path and no longer
        # pending.
        st = sess.status()
        assert st["save_path"] == str(saved)
        assert st["pending_review"] is False
        # Idempotent — a second commit returns the same path.
        assert sess.commit_playback() == str(saved)
    finally:
        sess.close()


# ── (3) discard removes draft, save_path never appears ───────────────


def test_discard_removes_draft_and_no_final_path(tmp_path):
    from openra_bench.human_labeling import InteractiveSession

    compiled = _smallest_easy_pack()
    assert compiled is not None

    sess = InteractiveSession(
        compiled, seed=1, record=True, playback_root=tmp_path,
        player="Human", manual_review=True,
    )
    try:
        _drive_to_done(sess)
        draft = Path(sess.status()["draft_path"])
        assert draft.exists()

        sess.discard_playback()
        # The draft dir is gone.
        assert not draft.exists()
        # No canonical save_path was ever published.
        assert sess.status()["save_path"] is None
        assert _final_seed_dir(tmp_path) is None
        # Regression: the per-session run dir (parent of `.draft/`)
        # must NOT be left behind as an empty carcass. The user-
        # facing bug was that Discard removed the files but left an
        # empty `run-<ts>__<player>/` folder visible to anyone
        # browsing the playback root.
        run_dirs = [
            p for p in tmp_path.iterdir()
            if p.is_dir() and p.name != ".draft"
        ]
        assert run_dirs == [], (
            "Discard left empty run dir(s) behind: " + repr(run_dirs)
        )
        # And the shared playback root itself is preserved (caller's
        # tmp_path must NOT be erased).
        assert tmp_path.exists()
        # Idempotent — a second discard is a no-op.
        sess.discard_playback()
        assert sess.status()["save_path"] is None
    finally:
        sess.close()


# ── Back-compat: manual_review=False keeps legacy direct-write ──────


def test_default_non_review_still_saves_directly(tmp_path):
    """Sanity check: an `InteractiveSession` with `manual_review=False`
    (the default used by `run_human_session` and scripted callers)
    still writes straight to the final path — no draft staging."""
    from openra_bench.human_labeling import InteractiveSession

    compiled = _smallest_easy_pack()
    assert compiled is not None

    sess = InteractiveSession(
        compiled, seed=1, record=True, playback_root=tmp_path,
        player="Human",  # manual_review defaults to False
    )
    try:
        _drive_to_done(sess)
        st = sess.status()
        assert st["manual_review"] is False
        assert st["pending_review"] is False
        save_path = st["save_path"]
        assert save_path, "non-review mode must publish save_path on done"
        saved = Path(save_path)
        assert ".draft" not in saved.parts
        assert (saved / "manifest.json").is_file()
    finally:
        sess.close()


# ── HTTP layer: commit / discard / preview endpoints ────────────────


def _load_game_api():
    """Import `site/game_api.py` as a module. The `site/` dir isn't a
    Python package (no __init__.py), so we go through importlib."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "openra_bench_site_game_api",
        Path(__file__).resolve().parents[1] / "site" / "game_api.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Spin up the FastAPI app with PLAYBACK_ROOT pointed at tmp."""
    pytest.importorskip("fastapi")
    starlette = pytest.importorskip("starlette.testclient")
    monkeypatch.setenv("OPENRA_BENCH_PLAYBACK_ROOT", str(tmp_path))

    game_api = _load_game_api()
    # Reset session store between tests so ids don't collide.
    game_api._sessions.clear()
    return starlette.TestClient(game_api.app), game_api, tmp_path


def _start_and_finish_via_api(client, pack_id):
    r = client.post(
        "/api/game/start",
        json={"pack_id": pack_id, "level": "easy", "seed": 1},
    )
    assert r.status_code == 200, r.text
    state = r.json()
    sid = state["session_id"]
    # Drive observe-only turns until the engine ends the episode.
    guard = state["max_turns"] + 5
    steps = 0
    while not state["done"] and steps < guard:
        r = client.post(
            "/api/game/step",
            json={"session_id": sid, "actions": []},
        )
        assert r.status_code == 200, r.text
        state = r.json()
        steps += 1
    assert state["done"]
    return sid, state


def test_api_commit_endpoint_promotes_draft(api_client):
    client, game_api, tmp_path = api_client
    compiled = _smallest_easy_pack()
    assert compiled is not None
    sid, state = _start_and_finish_via_api(client, compiled.pack_id)
    # The /step response carries the review-pending flags and NO
    # save_path yet.
    assert state["pending_review"] is True
    assert state["manual_review"] is True
    assert state["save_path"] in (None, "")

    # /preview returns the staged turns.
    r = client.get(f"/api/game/preview/{sid}")
    assert r.status_code == 200, r.text
    pv = r.json()
    assert pv["pending_review"] is True
    assert isinstance(pv["preview"], list)
    assert len(pv["preview"]) == state["turn"]

    # /commit promotes the draft and returns the final save_path.
    r = client.post(f"/api/game/commit/{sid}")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "committed"
    saved = Path(out["save_path"])
    assert saved.is_dir()
    assert (saved / "manifest.json").is_file()
    # And the session id is consumed.
    assert sid not in game_api._sessions


def test_api_discard_endpoint_removes_draft(api_client):
    client, game_api, tmp_path = api_client
    compiled = _smallest_easy_pack()
    assert compiled is not None
    sid, state = _start_and_finish_via_api(client, compiled.pack_id)
    draft = Path(state["draft_path"])
    assert draft.is_dir()

    r = client.post(f"/api/game/discard/{sid}")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "discarded"
    # Draft is gone, no canonical save dir was ever made.
    assert not draft.exists()
    assert _final_seed_dir(tmp_path) is None
    # Session id is consumed.
    assert sid not in game_api._sessions
