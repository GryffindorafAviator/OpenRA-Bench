# Vendored from OpenRA-RL-Training (byte-identical)

So the bench prompt/briefing/minimap format is **identical by
construction** to the training rollouts (no drift, no re-implementation):

| vendored file      | upstream source                                   |
|--------------------|---------------------------------------------------|
| `system_v2.txt`    | `openra_rl_training/prompts/system_v2.txt`        |
| `briefing_v2.py`   | `openra_rl_training/prompts/briefing_v2.py`       |
| `minimap_v2.py`    | `scripts/_minimap_v2.py`                           |

Do **not** hand-edit. `tests/test_vendor_drift.py` byte-compares these
against the upstream originals when the training checkout is present
(skips otherwise) so divergence is caught. To resync: re-copy and
re-run the suite.
