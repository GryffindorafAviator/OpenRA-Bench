"""Streamlit entrypoint for the playback viewer.

    pip install streamlit
    streamlit run scripts/view_playback.py -- <playback_root>
"""
import sys
from pathlib import Path

# Streamlit launches its own subprocess with a CWD/sys.path that doesn't
# include the repo root, so `openra_bench` can't be found by default.
# Inject the repo root (the parent of this script's dir) so the import
# works without the operator having to set PYTHONPATH externally.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openra_bench.playback_view import render_streamlit

if __name__ == "__main__":
    render_streamlit(sys.argv[1] if len(sys.argv) > 1 else "playback")
