"""Streamlit entrypoint for the playback viewer.

    pip install streamlit
    streamlit run scripts/view_playback.py -- <playback_root>
"""
import sys

from openra_bench.playback_view import render_streamlit

if __name__ == "__main__":
    render_streamlit(sys.argv[1] if len(sys.argv) > 1 else "playback")
