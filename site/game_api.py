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


def _prune_sessions():
    if len(_sessions) <= MAX_SESSIONS:
        return
    for sid in list(_sessions.keys())[: len(_sessions) - MAX_SESSIONS]:
        try:
            _sessions[sid].close()
        except Exception:
            pass
        del _sessions[sid]


# ── Request / response models ──────────────────────────────────────────────

class StartRequest(BaseModel):
    pack_id: str
    level: str = "easy"
    seed: int = 1


class ActionItem(BaseModel):
    mode: str  # "move" | "attack" | "observe" | "build" | "surrender"
    unit_ids: List[str] = []
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    target_id: Optional[str] = None
    building_type: Optional[str] = None


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
            })

    minimap_b64 = None
    try:
        from openra_bench.minimap import render_tactical_minimap
        img = render_tactical_minimap(rs, scale=4, grid=True, legend=True)
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
        "units": units,
        "enemies": enemies,
        "own_buildings": own_buildings,
        "minimap_b64": minimap_b64,
        "minimap_ascii": minimap_ascii,
        "explored_percent": round(sig.explored_percent, 2),
        "units_killed": sig.units_killed,
        "units_lost": sig.units_lost,
        "cash": sig.cash,
        "power_provided": sig.power_provided,
        "power_drained": sig.power_drained,
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
    try:
        from openra_bench.human_labeling import InteractiveSession

        sess = InteractiveSession.from_pack(
            req.pack_id, req.level, req.seed, record=True, player="Human",
        )
        sid = uuid.uuid4().hex[:12]
        sess._session_id = sid
        _sessions[sid] = sess
        return _serialize_state(sess)
    except Exception as e:
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
            ))

        sess.submit_turn(actions)
        return _serialize_state(sess)
    except Exception as e:
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


@app.get("/api/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


# ── Static file serving ────────────────────────────────────────────────────

SITE_DIR = Path(__file__).resolve().parent

@app.get("/")
def serve_index():
    return FileResponse(SITE_DIR / "index.html")


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
