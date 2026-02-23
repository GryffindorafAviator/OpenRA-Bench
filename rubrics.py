"""OpenRA-Bench rubrics — re-exported from openra-rl-util.

All scoring logic lives in the shared utility library.
This module re-exports for backward compatibility.
"""

from openra_rl_util.rubrics import (  # noqa: F401
    EconomyRubric,
    MilitaryEfficiencyRubric,
    OpenRABenchRubric,
    OpenRAWinLossRubric,
    compute_composite_score_from_games,
    compute_game_metrics,
)
