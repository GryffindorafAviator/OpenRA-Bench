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


# ── Error-path session cleanup (the "UI freeze" defect class) ────────
#
# These tests pin the invariant: every endpoint that LEAVES a sid
# unresponsive must ALSO remove it from `_sessions`. Without the
# defensive `_drop_session` on the error branches the session store
# leaked one corpse per failed step/commit/discard until the user
# restarted `python -m site.game_api` — the reported freeze.


def test_step_error_drops_session_from_store(api_client):
    """If submit_turn raises mid-step, the session MUST be evicted —
    otherwise its sid wedges a slot in `_sessions` forever and
    subsequent reqs against it 500 instead of 404 (user can't
    recover without restarting the server)."""
    client, game_api, _ = api_client
    compiled = _smallest_easy_pack()
    assert compiled is not None
    r = client.post(
        "/api/game/start",
        json={"pack_id": compiled.pack_id, "level": "easy", "seed": 1},
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid in game_api._sessions
    # Force submit_turn to raise by swapping it on the live session.
    sess = game_api._sessions[sid]

    def _boom(_actions):
        raise RuntimeError("simulated engine fault")

    sess.submit_turn = _boom  # type: ignore[method-assign]
    r = client.post(
        "/api/game/step",
        json={"session_id": sid, "actions": []},
    )
    assert r.status_code == 500
    # The sid is now gone — no zombie session left behind.
    assert sid not in game_api._sessions
    # And a follow-up request reports the (correct) 404 — NOT a
    # stale 500 from the same dead session.
    r2 = client.post(
        "/api/game/step",
        json={"session_id": sid, "actions": []},
    )
    assert r2.status_code == 404


def test_commit_failure_drops_session_from_store(api_client):
    """commit_playback() returning None (draft missing / already
    discarded) used to leak the sid in _sessions. The handler now
    drops it so the user can recover without restarting."""
    client, game_api, _ = api_client
    compiled = _smallest_easy_pack()
    assert compiled is not None
    sid, _state = _start_and_finish_via_api(client, compiled.pack_id)
    # Force commit to fail by pre-discarding the draft.
    sess = game_api._sessions[sid]
    sess.discard_playback()
    r = client.post(f"/api/game/commit/{sid}")
    assert r.status_code == 400
    # Crucial: even though commit failed, the sid is released.
    assert sid not in game_api._sessions


def test_sessions_debug_endpoint_lists_live_sids(api_client):
    """The /api/game/sessions debug endpoint lets the user inspect
    leaked sessions WITHOUT restarting the server — the primary
    self-recovery hook for the freeze symptom."""
    client, game_api, _ = api_client
    compiled = _smallest_easy_pack()
    assert compiled is not None
    r = client.post(
        "/api/game/start",
        json={"pack_id": compiled.pack_id, "level": "easy", "seed": 1},
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r = client.get("/api/game/sessions")
    assert r.status_code == 200
    payload = r.json()
    assert payload["count"] >= 1
    assert any(s["session_id"] == sid for s in payload["sessions"])

    # The /clear endpoint nukes every session — the "restart the
    # server" escape hatch, but without restarting the server.
    r = client.post("/api/game/sessions/clear")
    assert r.status_code == 200
    assert sid in r.json()["dropped"]
    assert sid not in game_api._sessions

    r = client.get("/api/game/sessions")
    assert r.json()["count"] == 0


def test_double_start_does_not_leak_first_session(api_client):
    """When the browser fires multiple Start requests (the click-
    spam race the frontend used to allow, or a long manual-play
    session that ran through many missions), the server-side store
    MUST eventually evict the oldest sessions — otherwise the
    store grows unbounded and every Start has to walk a longer
    dict, and pre-MAX leaks would accumulate.

    `_prune_sessions` is called at the TOP of start_game (before
    insertion), so the post-add steady state is ≤ MAX_SESSIONS + 1.
    """
    client, game_api, _ = api_client
    compiled = _smallest_easy_pack()
    assert compiled is not None
    # Fire MAX_SESSIONS + 3 starts. The oldest must have been
    # evicted; we don't assert on the precise eviction index
    # because the prune-before-insert ordering keeps one extra
    # corpse around briefly.
    sids = []
    for _ in range(game_api.MAX_SESSIONS + 3):
        r = client.post(
            "/api/game/start",
            json={"pack_id": compiled.pack_id, "level": "easy", "seed": 1},
        )
        assert r.status_code == 200
        sids.append(r.json()["session_id"])
    # The very oldest must be gone, the newest must be live, and
    # the store is bounded.
    assert sids[0] not in game_api._sessions
    assert sids[-1] in game_api._sessions
    assert len(game_api._sessions) <= game_api.MAX_SESSIONS + 1


def test_sessions_clear_destroys_unreviewed_draft(api_client, tmp_path):
    """Pin the data-loss surface of /api/game/sessions/clear (the
    backend half of the UI Refresh button).

    Refresh is the human's "the UI feels stuck — recover without
    server restart" escape hatch. Calling /clear drops every live
    session via InteractiveSession.close(), which for a manual-review
    session with an unreviewed draft falls into the abandoned-draft
    cleanup path (human_labeling.py:1207) — Playback.discard() runs
    and the draft directory is removed.

    This test pins that behaviour so future refactors don't
    accidentally LEAK the draft (occupying disk forever) OR PROMOTE
    it (data the user wanted to discard sneaking into the dataset).
    The frontend gameRefresh() guards against accidental loss via a
    confirm() dialog; this test just nails down what /clear itself
    does once the user has confirmed."""
    client, game_api, root = api_client
    compiled = _smallest_easy_pack()
    assert compiled is not None
    r = client.post(
        "/api/game/start",
        json={"pack_id": compiled.pack_id, "level": "easy", "seed": 1},
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]
    # Take one step so the session has a non-empty draft on disk.
    r = client.post(
        "/api/game/step",
        json={"session_id": sid, "actions": []},
    )
    assert r.status_code == 200
    # The session is still mid-game (not done) — its draft folder
    # exists under root/.draft/ but is not promoted. Sanity check.
    sess = game_api._sessions[sid]
    pb_dir = sess._playback.dir if sess._playback is not None else None
    assert pb_dir is not None
    assert pb_dir.exists(), "draft dir must exist after first step"

    # Refresh — clear all sessions.
    r = client.post("/api/game/sessions/clear")
    assert r.status_code == 200
    assert sid in r.json()["dropped"]
    assert sid not in game_api._sessions

    # The draft must be gone (Playback.discard() ran via close()'s
    # abandoned-draft cleanup path). Nothing was promoted — the final
    # save_path was never set.
    assert not pb_dir.exists(), "unreviewed draft must be discarded"
    # And no stray finalized cell directory landed at the real save
    # location — we're not silently promoting drafts on /clear.
    final_dirs = [
        p for p in root.rglob("seed*") if p.is_dir() and ".draft" not in p.parts
    ]
    assert not final_dirs, (
        f"/clear must not promote drafts; found {final_dirs}"
    )


# ── Production "ready vs building" surface (build+place UX bug) ──────
#
# A zh user reported "Build + Place 之后 map 上什么都没有" — clicked
# Build, end-turned until the UI said "ready", clicked Place, end-
# turned, and the new structure never appeared. Root cause: the bench
# adapter only carried the production item NAMES; the UI labelled
# every queued item as "ready" even when the engine still had
# `done: false`. Place clicks on a not-yet-done entry were silently
# rejected by the engine (`PLACE BLOCKED: <item> not completed in
# queue`) and the order was dropped.
#
# These tests pin the fix:
#   1. /api/game/step responses carry `production_detail` with
#      `{item, progress, done}` per queue entry.
#   2. A `build('proc')` queued without a paired sell on
#      build-sell-and-rebuild-elsewhere progresses but never reaches
#      `done: true` (the agent runs out of cash mid-build) — the UI
#      must therefore NEVER show this item as ready, only as a
#      partial-progress pending entry.
#   3. The canonical sell→build→place flow reaches `done: true`,
#      Place succeeds, and the new proc actually appears in
#      `own_buildings` at the requested cell.


def test_production_detail_surfaces_progress_and_done(api_client):
    """The /step response must carry per-entry production detail so
    the manual-play UI can distinguish "building" from "ready to
    place"."""
    client, _game_api, _ = api_client
    r = client.post(
        "/api/game/start",
        json={
            "pack_id": "build-sell-and-rebuild-elsewhere",
            "level": "easy",
            "seed": 1,
        },
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # build('proc') without selling — the engine accepts the order
    # and starts draining cash, but the entry will never reach done.
    r = client.post(
        "/api/game/step",
        json={
            "session_id": sid,
            "actions": [{"mode": "build", "building_type": "proc"}],
        },
    )
    assert r.status_code == 200
    state = r.json()
    detail = state.get("production_detail")
    assert isinstance(detail, list) and detail, (
        "production_detail must surface the in-progress entry"
    )
    entry = detail[0]
    assert entry["item"] == "proc"
    assert entry["done"] is False, (
        "queued-but-still-building entry must have done=False — the "
        "UI ready/pending split hinges on this"
    )
    assert 0.0 < entry["progress"] < 1.0, (
        f"in-progress entry must report 0<progress<1; got {entry}"
    )


def test_build_without_sell_never_reaches_done(api_client):
    """Cash-starved build on build-sell-and-rebuild-elsewhere never
    completes — `done` stays False all the way to cash=0 — so the UI
    must NEVER label `proc` as ready on this trace. Pins the
    invariant the user's bug report inverted (UI used to show
    `proc` as ready immediately after Build)."""
    client, _game_api, _ = api_client
    r = client.post(
        "/api/game/start",
        json={
            "pack_id": "build-sell-and-rebuild-elsewhere",
            "level": "easy",
            "seed": 1,
        },
    )
    sid = r.json()["session_id"]
    # Build without selling — the user's reported flow.
    client.post(
        "/api/game/step",
        json={
            "session_id": sid,
            "actions": [{"mode": "build", "building_type": "proc"}],
        },
    )
    saw_done = False
    for _ in range(15):
        r = client.post(
            "/api/game/step",
            json={"session_id": sid, "actions": []},
        )
        det = r.json().get("production_detail") or []
        if any(p.get("done") for p in det):
            saw_done = True
            break
    assert not saw_done, (
        "build('proc') WITHOUT a paired sell must NOT complete — the "
        "engine ran out of cash mid-build. If this fires the user's "
        "'Place silently no-op' bug is back: the UI would label the "
        "stuck entry as 'ready'."
    )


def test_sell_then_build_then_place_creates_new_building(api_client):
    """The canonical capability flow: SELL the forward proc, BUILD a
    new proc, wait for done=true, PLACE at the safe target cell. The
    new proc must materialise in own_buildings at the requested
    coords."""
    client, _game_api, _ = api_client
    r = client.post(
        "/api/game/start",
        json={
            "pack_id": "build-sell-and-rebuild-elsewhere",
            "level": "easy",
            "seed": 1,
        },
    )
    state = r.json()
    sid = state["session_id"]
    # Find the forward proc by type so we don't pin a specific id.
    forward_proc = next(
        b for b in state["own_buildings"]
        if b["type"] == "proc" and b["cell_x"] > 20
    )

    # Sell forward proc.
    r = client.post(
        "/api/game/step",
        json={
            "session_id": sid,
            "actions": [{"mode": "sell", "unit_ids": [forward_proc["id"]]}],
        },
    )
    assert r.status_code == 200
    # Build proc.
    r = client.post(
        "/api/game/step",
        json={
            "session_id": sid,
            "actions": [{"mode": "build", "building_type": "proc"}],
        },
    )
    assert r.status_code == 200
    # Step until done.
    done_seen = False
    for _ in range(20):
        r = client.post(
            "/api/game/step",
            json={"session_id": sid, "actions": []},
        )
        det = r.json().get("production_detail") or []
        if any(p.get("item") == "proc" and p.get("done") for p in det):
            done_seen = True
            break
    assert done_seen, "sell+build flow must reach done=True on proc"
    # Place at the safe target cell.
    r = client.post(
        "/api/game/step",
        json={
            "session_id": sid,
            "actions": [{
                "mode": "place_building",
                "building_type": "proc",
                "target_x": 16,
                "target_y": 8,
            }],
        },
    )
    state = r.json()
    safe_procs = [
        b for b in state["own_buildings"]
        if b["type"] == "proc" and b["cell_x"] < 20
    ]
    assert safe_procs, (
        "Place at (16,8) must materialise a new proc in own_buildings "
        "near the safe NW corner — the build+place flow's win path"
    )
