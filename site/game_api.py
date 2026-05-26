"""Game API server — serves the static site + REST endpoints for engine integration.

Wraps InteractiveSession to let the browser drive game sessions turn-by-turn.

Usage:
    python site/game_api.py                        # default port 8765
    python site/game_api.py --port 9000
    python site/game_api.py --host 0.0.0.0         # expose to network
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="OpenRA-Bench Game API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store ───────────────────────────────────────────────────────────

_sessions: dict[str, Any] = {}
MAX_SESSIONS = 8


def _drop_session(sid: str) -> None:
    """Pop a session from the store and close it best-effort. Never
    raises — callers use this on error paths where we must NOT mask
    the original exception with a cleanup failure."""
    sess = _sessions.pop(sid, None)
    if sess is None:
        return
    try:
        sess.close()
    except Exception:  # noqa: BLE001 — cleanup must not raise
        pass


def _prune_sessions():
    """First sweep dead/closed sessions; then, if still over cap,
    evict the oldest. Without the dead-sweep a half-closed session
    (e.g. a step that raised mid-flight) would occupy a slot
    forever, eventually filling the cap with corpses and making
    every new Start race a close()."""
    # Pass 1: drop sessions whose env is already released. Detection
    # is best-effort — InteractiveSession exposes a private `_closed`
    # flag, but we don't want to rely on it strictly.
    for sid in list(_sessions.keys()):
        sess = _sessions.get(sid)
        if sess is None:
            continue
        if getattr(sess, "_closed", False):
            _drop_session(sid)
    # Pass 2: cap by age (dict iteration order = insertion order).
    if len(_sessions) <= MAX_SESSIONS:
        return
    for sid in list(_sessions.keys())[: len(_sessions) - MAX_SESSIONS]:
        _drop_session(sid)


# ── Request / response models ──────────────────────────────────────────────

class StartRequest(BaseModel):
    pack_id: str
    level: str = "easy"
    seed: int = 1


class ActionItem(BaseModel):
    mode: str  # "move" | "attack" | "harvest" | "observe" | "build" | "place_building" | "surrender" | "power_down" | special target verbs
    unit_ids: List[str] = []
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    target_id: Optional[str] = None
    building_type: Optional[str] = None
    unit_type: Optional[str] = None
    item: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    actions: List[ActionItem] = []


# ── State serialization ────────────────────────────────────────────────────

def _serialize_state(sess) -> dict:
    """Convert render_state + session status to JSON-safe dict."""
    rs = sess.render_state()
    status = sess.status()
    sig = sess._adapter.signals

    units = []
    for u in rs.get("units_summary", []) or []:
        if not isinstance(u, dict):
            continue
        units.append({
            "id": str(u.get("id", "")),
            "type": u.get("type", "?"),
            "cell_x": u.get("cell_x", 0),
            "cell_y": u.get("cell_y", 0),
            "hp": round(float(u.get("hp", 1.0)), 2),
            "activity": u.get("activity"),
            "idle": u.get("idle", False),
        })

    enemies = []
    for e in rs.get("enemy_summary", []) or []:
        if not isinstance(e, dict):
            continue
        enemies.append({
            "id": str(e.get("id", "")),
            "type": e.get("type", "?"),
            "cell_x": e.get("cell_x", 0),
            "cell_y": e.get("cell_y", 0),
            "hp": round(float(e.get("hp", 1.0)), 2),
            "is_building": bool(e.get("is_building", False)),
        })

    own_buildings = []
    for b in rs.get("own_buildings", []) or []:
        if isinstance(b, dict):
            own_buildings.append({
                "id": str(b.get("id", "")),
                "type": b.get("type", "?"),
                "cell_x": b.get("cell_x", 0),
                "cell_y": b.get("cell_y", 0),
                "hp": round(float(b.get("hp", 1.0)), 2),
                "is_building": True,
            })

    live_resource_cells = []
    for p in rs.get("resource_cells", []) or []:
        if not isinstance(p, dict):
            continue
        try:
            live_resource_cells.append({
                "cell_x": int(p.get("cell_x", 0)),
                "cell_y": int(p.get("cell_y", 0)),
                "type": "ore",
            })
        except (TypeError, ValueError):
            continue
    # Prefer the engine's live ore cells when the spatial tensor is
    # available: those cells shrink/disappear as harvesters deplete them.
    # Older observations may not expose spatial resource channels, so
    # fall back to authored mine markers only in that case.
    spatial_shape = rs.get("spatial_shape", (0, 0, 0)) or (0, 0, 0)
    try:
        has_resource_channel = int(spatial_shape[2]) > 5
    except (TypeError, ValueError, IndexError):
        has_resource_channel = False
    harvest_points = live_resource_cells
    if not has_resource_channel:
        for p in rs.get("harvest_points", []) or []:
            if not isinstance(p, dict):
                continue
            try:
                harvest_points.append({
                    "cell_x": int(p.get("cell_x", 0)),
                    "cell_y": int(p.get("cell_y", 0)),
                    "type": p.get("type", "mine"),
                })
            except (TypeError, ValueError):
                continue

    minimap_b64 = None
    try:
        from openra_bench.minimap import render_tactical_minimap
        # Pass `base_map` so the renderer paints the water/wall terrain
        # underlay (channels, bridges, walls visible from t=0). Best-
        # effort: `sess.compiled.scenario.base_map` is the canonical
        # source, falling back to '' if the session lacks it.
        _bm = ""
        try:
            _bm = sess.compiled.scenario.base_map or ""
        except Exception:  # noqa: BLE001
            _bm = ""
        img = render_tactical_minimap(
            rs, scale=4, grid=True, legend=True, base_map=_bm,
        )
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            minimap_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    minimap_ascii = rs.get("minimap", "")

    return {
        "session_id": getattr(sess, "_session_id", ""),
        "turn": status["turn"],
        "max_turns": status["max_turns"],
        "outcome": status["outcome"],
        "done": status["done"],
        "game_tick": status.get("tick", sig.game_tick),
        "objective": sess.objective,
        "save_path": status.get("save_path"),
        "manual_review": status.get("manual_review", False),
        "pending_review": status.get("pending_review", False),
        "draft_path": status.get("draft_path"),
        "units": units,
        "enemies": enemies,
        "own_buildings": own_buildings,
        "harvest_points": harvest_points,
        "minimap_b64": minimap_b64,
        "minimap_ascii": minimap_ascii,
        # 3-state shroud cells for clients that want to overlay their
        # own fog (the PNG already bakes in the dim/bright distinction;
        # exposing the raw cell lists lets the frontend draw e.g. a
        # canvas overlay independent of the PNG resolution).
        "visible_cells": rs.get("visible_cells", []),
        "fogged_cells": rs.get("fogged_cells", []),
        "explored_percent": round(sig.explored_percent, 2),
        "units_killed": sig.units_killed,
        "units_lost": sig.units_lost,
        "cash": sig.cash,
        "resources": sig.resources,
        "resource_capacity": sig.resource_capacity,
        "economy_value": sig.cash + sig.resources,
        "harvesters": sig.harvesters,
        "power_provided": sig.power_provided,
        "power_drained": sig.power_drained,
        "production": list(rs.get("production", []) or []),
        # Per-entry detail (item / progress / done) so the UI can tell
        # "Ready (click Place)" apart from "Building (30%)". Without
        # this the manual-play UI labelled every queued item as
        # "ready", and Place silently no-op'd on a not-yet-done entry
        # because the engine rejects place_building until `done=true`.
        "production_detail": [
            dict(p) for p in (rs.get("production_detail", []) or [])
            if isinstance(p, dict)
        ],
        "available_production": list(rs.get("available_production", []) or []),
        "tools": list(
            getattr(sess.compiled, "tools", None)
            or getattr(getattr(sess.compiled, "scenario", None), "tools", None)
            or []
        ),
    }


# ── API endpoints ──────────────────────────────────────────────────────────

@app.get("/api/scenarios")
def list_scenarios():
    """List all playable scenario pack ids."""
    try:
        from openra_bench.scenarios import load_pack
        from openra_bench.scenarios.loader import PACKS_DIR

        out = []
        for f in sorted(PACKS_DIR.glob("*.yaml")):
            if f.name.startswith(("_", "TEMPLATE")):
                continue
            try:
                p = load_pack(f)
                if p.meta.status == "active":
                    out.append({
                        "id": p.meta.id,
                        "title": p.meta.title,
                        "capability": p.meta.capability,
                        "levels": ["easy", "medium", "hard"],
                    })
            except Exception:
                continue
        return {"scenarios": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/game/start")
def start_game(req: StartRequest):
    """Start a new game session. Returns initial state."""
    _prune_sessions()
    sess = None
    sid = None
    try:
        from openra_bench.human_labeling import InteractiveSession

        # The browser UI is the human-review-before-save path: stage
        # playback artifacts in `.draft/` until the human clicks
        # Save or Discard on the end-of-game review modal.
        sess = InteractiveSession.from_pack(
            req.pack_id, req.level, req.seed, record=True, player="Human",
            manual_review=True,
        )
        sid = uuid.uuid4().hex[:12]
        sess._session_id = sid
        _sessions[sid] = sess
        return _serialize_state(sess)
    except Exception as e:
        # If construction got far enough to create the session but
        # state-serialization failed, the partial session would
        # otherwise leak (open env, open turns.jsonl handle). Close
        # it explicitly so the slot — and the engine handle — are
        # released before we return 4xx.
        if sid is not None:
            _drop_session(sid)
        elif sess is not None:
            try:
                sess.close()
            except Exception:  # noqa: BLE001
                pass
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/game/step")
def step_game(req: StepRequest):
    """Submit actions and advance one turn."""
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess.done:
        return _serialize_state(sess)

    try:
        from openra_bench.human_labeling import HumanAction

        actions = []
        for a in req.actions:
            target = None
            if a.target_x is not None and a.target_y is not None:
                target = (a.target_x, a.target_y)
            actions.append(HumanAction(
                mode=a.mode,
                units=a.unit_ids,
                target=target,
                target_id=a.target_id,
                unit_type=a.unit_type or a.building_type or a.item or "",
            ))

        sess.submit_turn(actions)
        return _serialize_state(sess)
    except Exception as e:
        # The session is now likely in a half-broken state (engine
        # raised mid-step, or render_state followed an env shutdown).
        # Drop it from _sessions so subsequent requests on this sid
        # surface 404 instead of piling more errors onto a dead
        # session — and so the slot is freed for the next Start
        # rather than counting toward MAX_SESSIONS.
        _drop_session(req.session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/game/state/{session_id}")
def get_state(session_id: str):
    """Get current game state."""
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _serialize_state(sess)


@app.post("/api/game/stop/{session_id}")
def stop_game(session_id: str):
    """Close a game session."""
    sess = _sessions.pop(session_id, None)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        sess.close()
    except Exception:
        pass
    return {"status": "closed"}


@app.get("/api/game/preview/{session_id}")
def preview_game(session_id: str):
    """Return the per-turn summary of the staged (draft) playback so
    the UI can show the human exactly what would be saved if they
    click Save. Available once `done=true`."""
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "outcome": sess.outcome,
        "turns": sess.turn,
        "max_turns": sess.max_turns,
        "pending_review": getattr(sess, "_finalized", False)
        and getattr(sess, "_manual_review", False)
        and not getattr(sess, "_committed", False)
        and not getattr(sess, "_discarded", False),
        "draft_path": getattr(sess, "draft_path", None),
        "preview": sess.preview_turns(),
    }


@app.post("/api/game/commit/{session_id}")
def commit_game(session_id: str):
    """Promote the staged playback draft to the real playback root
    and return the published `save_path`. Idempotent. The session is
    closed afterwards (its env handle is released) so the browser
    should treat the session id as consumed."""
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not getattr(sess, "_manual_review", False):
        raise HTTPException(
            status_code=400,
            detail="Session is not in manual-review mode",
        )
    if not getattr(sess, "_finalized", False):
        raise HTTPException(
            status_code=400,
            detail="Session has not finished yet — nothing to commit",
        )
    try:
        save_path = sess.commit_playback()
    except Exception as e:
        # commit_playback() catches its own promote() failures, but
        # guard the call site too — any unexpected exception must
        # still release the session slot, not leak it.
        _drop_session(session_id)
        raise HTTPException(
            status_code=500, detail=f"Commit raised: {e}"
        )
    if save_path is None:
        # Nothing to promote (already-discarded / draft missing).
        # Drop the session so the user can start fresh rather than
        # being wedged on a sid that 400s forever.
        _drop_session(session_id)
        raise HTTPException(
            status_code=400,
            detail="Commit failed — draft missing or already discarded",
        )
    # Release the engine handle now that the draft has been promoted.
    _drop_session(session_id)
    return {"status": "committed", "save_path": save_path}


@app.post("/api/game/discard/{session_id}")
def discard_game(session_id: str):
    """Drop the staged playback draft without saving and close the
    session. Idempotent on already-discarded sessions."""
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not getattr(sess, "_manual_review", False):
        raise HTTPException(
            status_code=400,
            detail="Session is not in manual-review mode",
        )
    # discard_playback() already swallows its own exceptions (see
    # human_labeling.InteractiveSession.discard_playback), but wrap
    # the call defensively: the contract is that this endpoint MUST
    # free the session slot regardless of whether the rmtree
    # succeeded — otherwise a permission-denied / file-lock during
    # rmtree (Windows / SMB) would wedge the sid in _sessions.
    try:
        sess.discard_playback()
    except Exception:  # noqa: BLE001
        pass
    _drop_session(session_id)
    return {"status": "discarded"}


@app.get("/api/game/sessions")
def list_sessions():
    """Debug endpoint: enumerate live sessions so the user can see
    leaked sids without restarting the server. Returns each sid's
    turn count, done flag, manual-review/finalized/committed state.
    Useful when the UI freezes — `curl /api/game/sessions` to find
    out if the slot is stuck."""
    out = []
    for sid, sess in _sessions.items():
        try:
            out.append({
                "session_id": sid,
                "turn": getattr(sess, "turn", None),
                "max_turns": getattr(sess, "max_turns", None),
                "done": getattr(sess, "done", None),
                "outcome": getattr(sess, "outcome", None),
                "manual_review": getattr(sess, "_manual_review", False),
                "finalized": getattr(sess, "_finalized", False),
                "committed": getattr(sess, "_committed", False),
                "discarded": getattr(sess, "_discarded", False),
                "closed": getattr(sess, "_closed", False),
            })
        except Exception as e:  # noqa: BLE001
            out.append({"session_id": sid, "error": str(e)})
    return {"sessions": out, "count": len(out), "cap": MAX_SESSIONS}


@app.post("/api/game/sessions/clear")
def clear_sessions():
    """Debug endpoint: drop every live session and close it. Recover
    from a wedged store WITHOUT restarting `python -m site.game_api`
    — the user's primary complaint. Returns the list of dropped
    sids."""
    dropped = list(_sessions.keys())
    for sid in dropped:
        _drop_session(sid)
    return {"dropped": dropped, "count": len(dropped)}


@app.get("/api/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


# ── Static file serving ────────────────────────────────────────────────────

SITE_DIR = Path(__file__).resolve().parent

@app.get("/")
def serve_index():
    return FileResponse(SITE_DIR / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/public", StaticFiles(directory=str(SITE_DIR / "public")), name="public")


# ── CLI entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="OpenRA-Bench Game API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Game API: http://{args.host}:{args.port}")
    print(f"Static site: http://{args.host}:{args.port}/")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
