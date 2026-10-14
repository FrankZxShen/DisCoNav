"""Exploration/backtracking score and history-state utilities.

The implementation deliberately separates two quantities that were
previously conflated under the name ``HFOV``:

1. directional history evidence, used to rank history nodes later; and
2. the frontier/history switching rule, controlled directly by JS.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


NavigationMode = Literal["frontier", "history"]


def directional_history_evidence(
    exploration_score: float,
    judgment_score: float,
    tau_es: float = 2.0,
    tau_js: float = 1.0,
) -> float:
    """Return the linear evidence written into the current heading bins."""

    return float(tau_es * exploration_score + tau_js * judgment_score)


def should_explore_frontier(
    judgment_score: float,
    local_step: int,
    beta: float = 0.3,
    warmup_steps: int = 125,
) -> bool:
    """Apply the implemented warm-up and post-warm-up switching policy.

    Steps 0 through ``warmup_steps`` (inclusive) use frontier exploration.
    Afterward, the judgment score is compared directly with ``beta``.
    """

    return int(local_step) <= int(warmup_steps) or float(judgment_score) >= float(beta)


def navigation_mode(
    judgment_score: float,
    local_step: int,
    beta: float = 0.3,
    warmup_steps: int = 125,
) -> NavigationMode:
    """Return the proposal/execution mode from the shared switching rule."""

    return (
        "frontier"
        if should_explore_frontier(judgment_score, local_step, beta, warmup_steps)
        else "history"
    )


def circular_heading_indices(
    heading_degrees: float,
    half_width: int = 39,
    num_bins: int = 360,
) -> np.ndarray:
    """Return a fixed-width circular window around a heading.

    With the released setting ``half_width=39``, this returns 78 bins matching
    the non-wrapping legacy slice ``[c-39:c+39]``.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if half_width <= 0 or 2 * half_width > num_bins:
        raise ValueError("half_width must satisfy 0 < 2*half_width <= num_bins")

    center = int(heading_degrees) % num_bins
    return np.arange(center - half_width, center + half_width, dtype=int) % num_bins


def write_directional_evidence(
    history_state: np.ndarray,
    heading_degrees: float,
    evidence: float,
    half_width: int = 39,
) -> np.ndarray:
    """Overwrite the fixed circular heading window with current evidence."""

    if history_state.ndim != 1:
        raise ValueError("history_state must be one-dimensional")
    indices = circular_heading_indices(
        heading_degrees, half_width=half_width, num_bins=history_state.size
    )
    history_state[indices] = float(evidence)
    return history_state


def history_node_score(history_state: np.ndarray, visit_count: int) -> float:
    """Return the released node-ranking score with an explicit safe divisor."""

    if visit_count <= 0:
        raise ValueError("visit_count must be positive")
    return float(history_state.sum() / visit_count)
