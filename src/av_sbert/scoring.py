"""From cosine similarities to PAN answers, with unanswered problems.

A pair's raw score is the mean cosine similarity of its chunk pairs, in [-1, 1]. Two thresholds
``p1 < p2`` turn it into a PAN answer in [0, 1]:

* ``s <= p1``      -> rescaled linearly from [-1, p1] to [0, 0.49]  (different authors)
* ``p1 < s < p2``  -> exactly 0.5                                    (left unanswered)
* ``s >= p2``      -> rescaled linearly from [p2, 1] to [0.51, 1]    (same author)

Leaving the uncertain middle band unanswered is rewarded by c@1 and F0.5u. ``p1`` and ``p2`` are
chosen by grid search on held-out problems, maximising the PAN overall score. The submitted model used
p1 = 0.45 and p2 = 0.54.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence, Tuple

import numpy as np

from .metrics import evaluate_all


def rescale(value: float, orig_min: float, orig_max: float, new_min: float, new_max: float) -> float:
    span = orig_max - orig_min
    if span == 0:
        span = 1e-6
    return new_min + (float(value - orig_min) / span) * (new_max - new_min)


def correct_scores(scores: Iterable[float], p1: float, p2: float) -> np.ndarray:
    s = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores, dtype=float)
    low = 0.0 + (s - -1.0) / max(p1 - -1.0, 1e-6) * 0.49
    high = 0.51 + (s - p2) / max(1.0 - p2, 1e-6) * 0.49
    out = np.where(s <= p1, low, np.where(s < p2, 0.5, high))
    return np.clip(out, 0.0, 1.0)


def grid_search_p1p2(similarities: Sequence[float], labels: Sequence[int], step: float = 0.01,
                     low: float = 0.01, high: float = 0.99) -> Tuple[float, float, dict]:
    """Exhaustive search over ``p1 < p2`` on a grid; returns the best pair and its metrics."""
    grid = np.round(np.arange(low, high, step), 4)
    best, best_pair = None, (0.5, 0.5)
    sims = np.asarray(similarities, dtype=float)
    for p1, p2 in combinations(grid, 2):
        result = evaluate_all(labels, correct_scores(sims, p1, p2))
        if best is None or result["overall"] > best["overall"]:
            best, best_pair = result, (float(p1), float(p2))
    return best_pair[0], best_pair[1], best
